from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train domain Taxon NER model for RLPE.")
    parser.add_argument("--train-json", type=str, required=True, help="Path to token-level labeled data.")
    parser.add_argument("--base-model", type=str, default="distilbert-base-cased")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()

    from datasets import Dataset
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    with Path(args.train_json).open("r", encoding="utf-8") as f:
        rows = json.load(f)

    # rows format example:
    # [{"tokens":[...], "ner_tags":[...]}, ...]
    ds = Dataset.from_list(rows)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # label map from data
    labels = sorted({tag for r in rows for tag in r["ner_tags"]})
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    def tokenize_and_align(examples):
        tokenized = tokenizer(examples["tokens"], is_split_into_words=True, truncation=True)
        all_labels = []
        for i, tags in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            label_ids = []
            prev = None
            for wid in word_ids:
                if wid is None:
                    label_ids.append(-100)
                elif wid != prev:
                    label_ids.append(label2id[tags[wid]])
                else:
                    label_ids.append(label2id[tags[wid]])
                prev = wid
            all_labels.append(label_ids)
        tokenized["labels"] = all_labels
        return tokenized

    ds = ds.map(tokenize_and_align, batched=True)

    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_steps=20,
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"saved model to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
