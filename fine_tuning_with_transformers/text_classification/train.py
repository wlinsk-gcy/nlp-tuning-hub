from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import evaluate
import numpy as np

task = "cola"
model_checkpoint = "distilbert-base-uncased"
batch_size = 16
# 除了mnli-mm以外，其他任务都可以直接通过任务名字进行加载。数据加载之后会自动缓存。
actual_task = "mnli" if task == "mnli-mm" else task
dataset = load_dataset("glue", actual_task)
metric = evaluate.load("glue", actual_task)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

# 数据预处理
# 不同的任务数据集的字段是不同的
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2")
}
sentence1_key, sentence2_key = task_to_keys[task]


def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)  # truncation 超过512自动截断
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)


# 当预处理函数处理多个样本时, 输出的一个list结构; 需要通过map操作映射会dataset的格式
# 对数据集datasets里面的所有样本进行预处理，使用map函数将预处理函数应用到所有的样本上
encoded_dataset = dataset.map(preprocess_function, batched=True, num_proc=4)

num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"

args = TrainingArguments(
    "test-glue",
    eval_strategy="epoch",  # 每轮都测评
    save_strategy="epoch",  # 每轮都保存权重
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    report_to="none"  # 关闭wandb等
)


# 不同的任务，测评指标不同；
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)


validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

if __name__ == '__main__':
    trainer.train()
