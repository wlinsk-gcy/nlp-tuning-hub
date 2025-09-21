from datasets import load_dataset
from evaluate import load
import nltk
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, \
    Seq2SeqTrainer

model_checkpoint = "t5-small"
model_name = model_checkpoint.split("/")[-1]

if model_checkpoint in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]:
    prefix = "summarize: "
else:
    prefix = ""

max_input_length = 1024
max_target_length = 128
batch_size = 16


# 加载数据集
raw_datasets = load_dataset("xsum")
# 加载评估指标
metric = load("rouge")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# 数据预处理函数
def preprocess_function(examples):
    # 添加样本前缀
    inputs = [prefix + doc for doc in examples["document"]]
    # 当样本长度超过max_length，由于truncation=True，所以会自动发生截断
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # 在seq2seq模型中，source text是用于encoder，target text是用于decoder在训练时需要预测的标签
    # 所以这里会选择把summary转成input_ids组成的labels，用于decoder预测
    # 如果只是简单的tokenizer(examples["summary"])，分词器会认为该样本只是普通数据，不知道这是训练的目标；
    labels = tokenizer(text_target=examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

# 加载预训练模型
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-xsum",
    eval_strategy="epoch",  # 每个epoch训练完即评估
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,  # seq2seqTrainer会定期保存模型权重，所以设置最多保存3个
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=True,  # 混合精度
    report_to="none"
)

# 它会将每个batch的input进行最大长度对齐，同样也会将labels填充到最大长度；
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# 封装评估指标计算函数
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # 把labels中-100的值，替换成分词器的特殊token，pad_token，在decode时可以跳过
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # ROUGE 指标库希望输入的参考答案和预测答案是 按句子分行的文本。
    # nltk.sent_tokenize：会把一句话切分成多个句子。例如："I love NLP. It is fun." → ["I love NLP.", "It is fun."]
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    # use_stemmer=True：主要是把单词提取词干，例如：running → run，避免因为词形变化导致分数偏低。
    # use_aggregator=True：会对整个 dataset 的结果聚合（平均），而不是逐个样本输出
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)
    # 把结果转成百分制
    result = {key: value * 100 for key, value in result.items()}

    # 统计predictions里面非特殊token的数量，再取平均值
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


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
    # 开始训练
    trainer.train()
