# taken from https://github.com/huggingface/cosmopedia/blob/main/classification/train_edu_bert.py


from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
)
from datasets import load_dataset, ClassLabel
import datasets
import numpy as np
import evaluate
import argparse
import os
from sklearn.metrics import classification_report, confusion_matrix


def compute_metrics(eval_pred):
    print("=======")
    print("Computing metrics...")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = eval_pred
    preds = np.round(logits.squeeze()).clip(0, 5).astype(int)
    labels = np.round(labels.squeeze()).astype(int)
    precision = precision_metric.compute(
        predictions=preds, references=labels, average="macro"
    )["precision"]
    recall = recall_metric.compute(
        predictions=preds, references=labels, average="macro"
    )["recall"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]

    report = classification_report(labels, preds)
    cm = confusion_matrix(labels, preds)
    print("Validation Report:\n" + report)
    print("Confusion Matrix:\n" + str(cm))

    return {
        "precision": precision,
        "recall": recall,
        "f1_macro": f1,
        "accuracy": accuracy,
    }


def main(args):
    dataset = load_dataset("lapa-llm/voxcheck-disinfo-qas", split="train")
    dataset
    # %%
    dataset[0]

    # %%
    # strip tags using beautifulsoup
    from bs4 import BeautifulSoup

    s = BeautifulSoup(dataset[0]["debunking"], "html.parser")
    print(s.get_text())

    dataset = dataset.map(
        lambda x: {
            "debunking": BeautifulSoup(x["debunking"], "html.parser").get_text()
        },
        num_proc=16,
    )

    dataset = dataset.train_test_split(train_size=0.9, seed=42)

    def convert_dataset_to_labels(dataset):
        propaganda = []
        debunking = []
        for example in dataset:
            # propaganda.append(example["fakes_thesis"])
            propaganda.append(example["propaganda_uncensored_answer"])
            propaganda.append(example["attack_debunking_propaganda_answer"])
            debunking.append(example["debunking"])
            debunking.append(example["factual_answer"])
            debunking.append(example["factual_debunking_answer"])
        propaganda = list(set(propaganda))
        debunking = list(set(debunking))
        propaganda = [(p, 0) for p in propaganda]
        debunking = [(d, 1) for d in debunking]
        new_dataset = propaganda + debunking
        new_dataset = datasets.Dataset.from_dict(
            {"text": [x[0] for x in new_dataset], "score": [x[1] for x in new_dataset]}
        )
        return new_dataset

    train_dataset = convert_dataset_to_labels(dataset["train"])
    test_dataset = convert_dataset_to_labels(dataset["test"])

    propaganda_dataset = datasets.load_dataset(
        "lapa-llm/propaganda_dataset", split="test"
    )

    propaganda_dataset = propaganda_dataset.train_test_split(train_size=0.9, seed=42)

    def convert_propaganda_dataset(dataset):
        propaganda = []
        debunking = []
        for example in dataset:
            # propaganda.append(example["fakes_thesis"])
            propaganda.append(example["propaganda_answer"])
            debunking.append(example["factual_answer"])
        propaganda = list(set(propaganda))
        debunking = list(set(debunking))
        propaganda = [(p, 0) for p in propaganda]
        debunking = [(d, 1) for d in debunking]
        new_dataset = propaganda + debunking
        new_dataset = datasets.Dataset.from_dict(
            {"text": [x[0] for x in new_dataset], "score": [x[1] for x in new_dataset]}
        )
        return new_dataset

    train_propaganda_dataset = convert_propaganda_dataset(propaganda_dataset["train"])
    test_propaganda_dataset = convert_propaganda_dataset(propaganda_dataset["test"])

    total_train_dataset = datasets.concatenate_datasets(
        [train_dataset, train_propaganda_dataset]
    )
    total_test_dataset = datasets.concatenate_datasets(
        [test_dataset, test_propaganda_dataset]
    )

    dataset = datasets.DatasetDict(
        {"train": total_train_dataset, "test": total_test_dataset}
    )

    dataset = dataset.cast_column(
        args.target_column, ClassLabel(names=[str(i) for i in range(6)])
    )
    # dataset = dataset.train_test_split(
    #    train_size=0.9, seed=42, stratify_by_column=args.target_column
    # )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model_name,
        num_labels=1,
        classifier_dropout=0.0,
        hidden_dropout_prob=0.0,
        output_hidden_states=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name,
        model_max_length=min(model.config.max_position_embeddings, 512),
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess(examples):
        batch = tokenizer(examples["text"], truncation=True)
        batch["labels"] = np.float32(examples[args.target_column])
        return batch

    dataset = dataset.map(preprocess, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    for param in model.roberta.embeddings.parameters():
        param.requires_grad = False
    for param in model.roberta.encoder.parameters():
        param.requires_grad = False

    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir,
        hub_model_id=args.output_model_name,
        # eval_strategy="steps",
        # save_strategy="steps",
        # eval_steps=50,
        # save_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        eval_steps=1,
        save_steps=1,
        logging_steps=100,
        learning_rate=3e-6,
        num_train_epochs=6,
        warmup_steps=40,
        seed=0,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=128,
        eval_on_start=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        bf16=True,
        push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(os.path.join(args.checkpoint_dir, "final"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_name", type=str, default="intfloat/multilingual-e5-base"
    )
    parser.add_argument("--target_column", type=str, default="score")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./alignment_score_checkpoints",
    )
    parser.add_argument(
        "--output_model_name", type=str, default="lapa-llm/alignment-score-model"
    )
    args = parser.parse_args()

    main(args)
