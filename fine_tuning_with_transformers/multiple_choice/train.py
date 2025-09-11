from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer
from datasets import load_dataset
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch
import numpy as np

model_checkpoint = "bert-base-uncased"
batch_size = 16
ending_names = ["ending0", "ending1", "ending2", "ending3"]

datasets = load_dataset("swag", "regular")

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

model = AutoModelForMultipleChoice.from_pretrained(model_checkpoint)


def preprocess_function(examples):
    # 将题目重复4次，用来匹配每一个选项
    first_sentences = [[context] * 4 for context in examples["sent1"]]
    # 获取选项头
    question_headers = examples["sent2"]
    # 把每个选项头对应的4个结尾，拼接到选项头上
    second_sentences = [[f"{header} {examples[end][i]}" for end in ending_names] for i, header in
                        enumerate(question_headers)]

    # 将二维的展开成一维结构
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    # 把每个分词器的输出重新分组，每4条数据对应一个题目的4个选项
    # 保证模型输入是(batch_size,num_choice,seq_len)的形式
    return {k: [v[i:i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}


encoded_datasets = datasets.map(preprocess_function, batched=True)


@dataclass
class DataCollatorForMultipleChoice:
    """
    用于批处理多选数据时动态填充序列的数据收集器；
    padding默认为True，每个序列都会填充到当前批次中，选项的最大长度
    """
    tokenizer: PreTrainedTokenizerBase  # 传入的分词器，用于动态填充序列
    padding: Union[bool, str, PaddingStrategy] = True  # 是否填充（True 自动按最长序列，或指定 "max_length"）
    max_length: Optional[int] = None  # 填充/截断到的最大长度
    pad_to_multiple_of: Optional[int] = None  # 可选，将序列填充到某个倍数长度（比如 GPU Tensor Core 需要 8 的倍数）

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        # 提取所有样本的标签，并从 features 中删除标签字段（避免干扰后续 padding）
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        # 每个样本的选项数
        num_choices = len(features[0]["input_ids"])
        # 每条数据按照选项数展开
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in
                              features]
        # 展开成一维
        flattened_features = sum(flattened_features, [])
        # 对所有展开后的序列进行动态填充，返回 PyTorch tensor 格式
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        # 将填充好的 tensor 重新 reshape 回 (batch_size, num_choices, seq_len)，还原多选的结构
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # 将提取的标签加回去，作为最终返回的 batch
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


args = TrainingArguments(
    "test-swag",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    report_to="none"  # 关闭wandb等
)


def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}


trainer = Trainer(
    model,
    args,
    train_dataset=encoded_datasets["train"],
    eval_dataset=encoded_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer),
    compute_metrics=compute_metrics,
)

if __name__ == '__main__':
    trainer.train()