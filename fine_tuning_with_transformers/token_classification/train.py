from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import load_dataset
import evaluate
import numpy as np


task = "ner" #需要是"ner", "pos" 或者 "chunk"
model_checkpoint = "distilbert-base-uncased"
batch_size = 16

datasets = load_dataset("conll2003")

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

label_all_tokens = True

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # 把特殊token的id设置成-100，在训练阶段计算损失值时会自动忽略
            if word_idx is None:
                label_ids.append(-100)
            # 我们为每个单词的第一个token设置标签。
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])

            # 对于单词中的其他token，我们将标签设置为当前标签或 -100，具体取决于label_all_tokens的配置。
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

label_list = datasets["train"].features[f"{task}_tags"].feature.names
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))



args = TrainingArguments(
    f"test-{task}",
    eval_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    report_to="none"  # 关闭wandb等
)

data_collator = DataCollatorForTokenClassification(tokenizer)
metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }



trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

if __name__ == '__main__':
    trainer.train()