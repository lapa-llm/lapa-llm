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
    dataset = load_dataset("peterua/OmniGEC-ModelTraining", split="train")
    dataset = dataset.filter(lambda x: x["language"] == "ukrainian")
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    def create_new_dataset(dataset):
        incorrect_sentences = []
        correct_sentences = []
        for item in dataset:
            incorrect_sentences.append(item["text"])
            correct_sentences.append(item["correction"])

        incorrect_sentences = [
            {"text": sent, "gec-score": 0} for sent in incorrect_sentences
        ]
        correct_sentences = [
            {"text": sent, "gec-score": 1} for sent in correct_sentences
        ]
        return datasets.Dataset.from_dict(
            {
                "text": [x["text"] for x in incorrect_sentences + correct_sentences],
                "gec-score": [
                    x["gec-score"] for x in incorrect_sentences + correct_sentences
                ],
            }
        )

    dataset["train"] = create_new_dataset(dataset["train"])
    dataset["test"] = create_new_dataset(dataset["test"])

    dataset["train"] = dataset["train"].filter(
        lambda x: x["text"] is not None and len(x["text"].strip()) > 0
    )

    dataset["test"] = dataset["test"].filter(
        lambda x: x["text"] is not None and len(x["text"].strip()) > 0
    )

    dataset = dataset.cast_column(
        args.target_column, ClassLabel(names=[str(i) for i in range(2)])
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
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=400,
        save_steps=400,
        logging_steps=100,
        learning_rate=3e-5,
        num_train_epochs=60,
        lr_scheduler_type="cosine",
        warmup_steps=500,
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
    parser.add_argument("--target_column", type=str, default="gec-score")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./gec_score_checkpoints",
    )
    parser.add_argument(
        "--output_model_name", type=str, default="lapa-llm/gec-score-model"
    )
    args = parser.parse_args()

    main(args)
