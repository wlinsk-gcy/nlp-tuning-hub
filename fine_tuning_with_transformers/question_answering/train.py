from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, default_data_collator
from datasets import load_dataset
import collections
from tqdm.auto import tqdm
import evaluate
import numpy as np

# squad_v2等于True或者False分别代表使用SQUAD v1 或者 SQUAD v2。
# 如果您使用的是其他数据集，那么True代表的是：模型可以回答“不可回答”问题，也就是部分问题不给出答案，而False则代表所有问题必须回答。
squad_v2 = False
model_checkpoint = "distilbert-base-uncased"
batch_size = 16

max_length = 384  # 输入feature的最大长度，question和context拼接之后
doc_stride = 128  # 2个切片之间的重合token数

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
datasets = load_dataset("squad_v2" if squad_v2 else "squad")

pad_on_right = tokenizer.padding_side == "right"  # context在右边


def prepare_train_features(examples):
    # 既要对examples进行truncation（截断）和padding（补全）还要还要保留所有信息，所以要用的切片的方法。
    # 每一个超长文本example会被切片成多个输入，相邻两个输入之间会有交集。
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    # 我们使用overflow_to_sample_mapping参数来映射切片片ID到原始ID。
    # 比如有2个expamples被切成4片，那么对应是[0, 0, 1, 1]，前两片对应原来的第一个example。
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # offset_mapping也对应4片
    # offset_mapping参数帮助我们映射到原始输入，由于答案标注在原始输入上，所以有助于我们找到答案的起始和结束位置。
    offset_mapping = tokenized_examples.pop("offset_mapping")
    # 重新标注数据
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # 对每一片进行处理，将无答案的样本标注到CLS上
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        # 区分question和context
        sequence_ids = tokenized_examples.sequence_ids(i)
        # 拿到原始的example 下标.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # 如果没有答案，则使用CLS所在的位置为答案.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # 答案的character级别Start/end位置.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            # 找到token级别的index start.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1
            # 找到token级别的index end.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1
            # 检测答案是否超出文本长度，超出的话也用CLS index作为标注.
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # 如果不超出则找到答案token的start和end位置。.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


tokenized_datasets = datasets.map(prepare_train_features, batched=True, remove_columns=datasets["train"].column_names)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
args = TrainingArguments(
    f"test-squad",
    eval_strategy="epoch",
    learning_rate=2e-5,  # 学习率
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,  # 训练的论次
    weight_decay=0.01,
    report_to="none"
)
# 将预处理好的数据喂给模型
data_collator = default_data_collator

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

"""
评估阶段的处理函数如下：
"""
metric = evaluate.load("squad_v2" if squad_v2 else "squad")

def prepare_validation_features(examples):
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def postprocess_qa_predictions(examples, features, raw_predictions, tokenizer, n_best_size=20, max_answer_length=30):
    all_start_logits, all_end_logits = raw_predictions
    # 把原本的id list转成 kv形式方便定位每个example
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    # 用来存储每个 example 对应的 feature 索引（因为长文本会被切成多个特征）。
    features_per_example = collections.defaultdict(list)
    # 一个example可能对应多个特征集，所以要把多个特征集都归到同一个example下
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # 有序字典，用于存储每个样本的预测答案
    predictions = collections.OrderedDict()
    print(f"Post-processing {len(examples)} examples split into {len(features)} features.")

    for example_index in tqdm(range(len(examples))):
        example = examples[example_index]
        # 获取当前样本的特征集
        feature_indices = features_per_example[example_index]
        # 如果是 squad_v2，用于记录预测为 "无答案" 的得分（CLS token 分数）
        min_null_score = None
        # 保存当前样本的所有候选答案
        valid_answers = []
        context = example["context"]

        for feature_index in feature_indices:
            # 取出该 feature 对应的预测分数和 offset_mapping（token 到原文字符位置的映射）
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # 分词后的token位置和原始token的对应位置
            offset_mapping = features[feature_index]["offset_mapping"]

            # 找出 [CLS] 位置（模型用它表示无答案）
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            # 计算无答案得分（CLS 起点 + CLS 终点）
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            # 找当前样本所有切片中的最低无答案得分
            if min_null_score is None or feature_null_score < min_null_score:
                min_null_score = feature_null_score
            # 去分数最高的n_best_size个起点/终点，例如：
            start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()

            # 这里是双层for遍历所有组合
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # 超出上下文范围、offset 为 None、终点在起点前、答案长度超限的，都不要
                    if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                    ):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    # 记录合法的答案
                    valid_answers.append({
                        "score": start_logits[start_index] + end_logits[end_index],
                        "text": context[start_char:end_char],
                    })
        # 排序后取出最高分的答案
        best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0] if valid_answers else {
            "text": "", "score": 0.0}
        if not squad_v2:
            predictions[example["id"]] = best_answer["text"]
        else:
            predictions[example["id"]] = best_answer["text"] if best_answer["score"] > min_null_score else ""

    return predictions


if __name__ == '__main__':
    trainer.train()

    # 评估
    validation_features = datasets["validation"].map(
        prepare_validation_features,
        batched=True,
        remove_columns=datasets["validation"].column_names
    )
    # 获得所有预测结果
    raw_predictions = trainer.predict(validation_features)
    final_predictions = postprocess_qa_predictions(datasets["validation"], validation_features,
                                                   raw_predictions.predictions)

    # 对评估内容做一下格式化
    if squad_v2:
        formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in
                                 final_predictions.items()]
    else:
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"]]
    metric.compute(predictions=formatted_predictions, references=references)
