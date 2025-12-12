from typing import Dict, List, Tuple

import pandas as pd #librariews
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score # to evaluate

from ..config import HF_MODEL_NAME, RANDOM_STATE

# expected by trainer 
class PolarityDataset(Dataset):
    """
    Simple torch Dataset wrapping texts + labels for DistilBERT.
    """

    def __init__(self, texts, labels, tokenizer, label2id, max_length: int = 128):
        self.texts = list(texts)
        self.labels = [label2id[l] for l in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int: # for how many samples exist
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]: # to tokenize raw string and to apply truncation, padding
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
# to return tensors: supervised learning

def compute_metrics(eval_pred) -> Dict[str, float]: # to turn outputs of raw models, labels, into metric numbers.
    """
    Metrics callback for HuggingFace Trainer.
    Returns accuracy + weighted F1.
    """
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")

    return {"accuracy": acc, "f1_weighted": f1}


def _prepare_label_mappings(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    labels = sorted(df["label"].unique()) 
    label2id = {lab: i for i, lab in enumerate(labels)} # training
    id2label = {i: lab for lab, i in label2id.items()} # to have predictions be mapped back to
    return label2id, id2label

# to encompass the DistilBERT training as well as the tuning process
def train_neural_model(
    train_df: pd.DataFrame, test_df: pd.DataFrame
):
    """
    Fine-tune DistilBERT (HF_MODEL_NAME) on the training data with a small
    hyperparameter search over learning rate, batch size, and epochs. 

    Returns
    -------
    final_model : AutoModelForSequenceClassification
        DistilBERT model trained on the full dataset (train + test)
        using the best hyperparameters.
    tokenizer : AutoTokenizer
    label2id : dict
    id2label : dict
    best_metrics : dict
        Metrics on the validation (test) split for the best config,
        plus the config itself under key "best_config".
    """
    print("\n=== DistilBERT fine-tuning with hyperparameter search ===") # print!

    device = "cuda" if torch.cuda.is_available() else "cpu" # remember to select gpu or else the test would use the cpu.
    print(f"[NEURAL] Using device: {device}")
    # the following is to tokenize text consistently.
    # label mappings
    label2id, id2label = _prepare_label_mappings(train_df)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    # Next is for setting up two datasets, for training and for validation (evaluation/eval)
    # datasets
    train_dataset = PolarityDataset(
        train_df["text"], train_df["label"], tokenizer, label2id
    )
    eval_dataset = PolarityDataset(
        test_df["text"], test_df["label"], tokenizer, label2id
    )

    # to ensure reproducibility, approximately ensuring the same metric for re-runs.
    torch.manual_seed(RANDOM_STATE)
    if device == "cuda":
        torch.cuda.manual_seed_all(RANDOM_STATE)

    # hyperparameter configs to try -- to save time
    configs: List[Dict] = [
        {"learning_rate": 2e-5, "batch_size": 16, "epochs": 3},
        {"learning_rate": 3e-5, "batch_size": 16, "epochs": 4},
        {"learning_rate": 5e-5, "batch_size": 32, "epochs": 3},
    ]

    best_metrics = None
    best_config = None

    # hyperparameter 
    
    for idx, cfg in enumerate(configs, start=1):
        print(f"\n[NEURAL] Running config {idx}/{len(configs)}: {cfg}") # to try each config seperately/one by one.

        model = AutoModelForSequenceClassification.from_pretrained( # to avoid previous fine tuning and to start from identical pre-trained checkpoints (DistilBERT)
            HF_MODEL_NAME,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
        ).to(device) 

        args = TrainingArguments( # to set u trainer on where/how to save, evaluate, learn, log.
            output_dir=f"distilbert_search_run_{idx}",
            eval_strategy="epoch",
            save_strategy="no",
            learning_rate=cfg["learning_rate"],
            per_device_train_batch_size=cfg["batch_size"],
            per_device_eval_batch_size=cfg["batch_size"],
            num_train_epochs=cfg["epochs"],
            weight_decay=0.01,
            logging_steps=50,
            load_best_model_at_end=False,
            report_to=[],  # disable wandb etc.
            seed=RANDOM_STATE,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        metrics = trainer.evaluate()
        print(f"[NEURAL] Validation metrics for config {cfg}: {metrics}")

        # HuggingFace prefixes our metrics with "eval_"
        f1_key = "eval_f1_weighted"

        if best_metrics is None or metrics.get(f1_key, 0.0) > best_metrics.get(
            f1_key, 0.0
        ):
            best_metrics = metrics
            best_config = cfg

    print("\n[NEURAL] Best configuration found:") # print results!
    print(f"  {best_config}")
    print("[NEURAL] Best validation metrics:")
    print(best_metrics)

    
    # to re-train best config on the full dataset
  
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_dataset = PolarityDataset(
        full_df["text"], full_df["label"], tokenizer, label2id
    )

    print("\n[NEURAL] Re-training best configuration on full dataset...") # to do (and print!) after the best hyperparameters have been chosen
    final_model = AutoModelForSequenceClassification.from_pretrained(
        HF_MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    final_args = TrainingArguments( # to re-apply -- via the winning hyperparameters -- on the full dataset foor training
        output_dir="distilbert_final",
        eval_strategy="no",
        save_strategy="no",
        learning_rate=best_config["learning_rate"],
        per_device_train_batch_size=best_config["batch_size"],
        num_train_epochs=best_config["epochs"],
        weight_decay=0.01,
        logging_steps=50,
        report_to=[],
        seed=RANDOM_STATE,
    )

    final_trainer = Trainer(
        model=final_model,
        args=final_args,
        train_dataset=full_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    final_trainer.train()

    # Keep the best config info alongside metrics
    best_metrics = dict(best_metrics)
    best_metrics["best_config"] = best_config

    print("[NEURAL] Training complete. Returning final model and metrics.")
    return final_model, tokenizer, label2id, id2label, best_metrics
