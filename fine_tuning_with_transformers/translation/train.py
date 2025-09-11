from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, \
    Seq2SeqTrainer
import numpy as np
import evaluate

batch_size = 16
model_checkpoint = "Helsinki-NLP/opus-mt-en-ro"
model_name = model_checkpoint.split("/")[-1]

# 加载数据集
raw_datasets = load_dataset("wmt16", "ro-en")  # 罗马尼亚语 <-> 英语
metric = evaluate.load("sacrebleu")

# 分词器
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# 预训练模型
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# 如果你使用的是t5系列模型，需要在数据样本前面添加特殊前缀：
if model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
    prefix = "translate English to Romanian: "
else:
    prefix = ""

# 主要看数据集中样本的集中长度，选一个能覆盖大部分的即可，Helsinki-NLP/opus-mt-en-ro通常是128/256/512
max_input_length = 128  # 源样本的最大长度
max_target_length = 128  # 目标样本的最大长度
source_lang = "en"
target_lang = "ro"


def preprocess_function(examples):
    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # as_target_tokenizer： 在Seq2Seq任务中，告诉分词器此时处理的是目标样本，而不是源样本
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = raw_datasets.map(preprocess_function, batched=True, num_proc=4)

# 封装训练参数配置器
args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-{source_lang}-to-{target_lang}",  # 保存目录的名称，
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,  # 因为数据量比较大，设置最多保存三次即可，超过会自动删除最旧的。
    num_train_epochs=1,  # 只跑1轮是为了验证流程
    predict_with_generate=True,  # 在验证时自动生成预测结果，而不仅仅计算loss，对于翻译/摘要这种 Seq2Seq 任务必须开启
    fp16=True,  # 开启半精度，可以减少显存占用，加快训练速度
    report_to="none"
)

# 配置Seq2Seq数据加载器，它不仅会将输入填充到批次中的最大长度，还会将标签填充到最大长度
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# 封装后置处理函数
def postprocess_text(preds, labels):
    """Hugging Face 的 metric.compute 要求 references 的格式是List[List]（因为每个预测可能有多个参考翻译）"""
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


# 封装评估函数
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)  # 把token Id转回token

    # 把-100的标签，替换成pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)  # 把token Id转回token

    # 做后置处理
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}  # bleu: [0,100], 越高越好

    # 统计每个句子的不计pad_token的实际长度，主要是用来观察是否生成太短或太长的句子
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)

    result = {k: round(v, 4) for k, v in result.items()}
    return result


# 创建训练器
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

if __name__ == '__main__':
    # 在T4的卡下，一个epoch跑了1h
    trainer.train()
