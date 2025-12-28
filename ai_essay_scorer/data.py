from io import StringIO

import dvc.api
import pandas as pd
import pytorch_lightning as pl
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class LightningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_file: str,
        train_batch_size: int,
        val_and_test_batch_size: int,
        max_len: int,
        val_size: float,
        test_size: float,
        num_workers: int,
        seed: int,
        tokenizer_name=None,
    ):
        super().__init__()
        self.data_file = data_file
        self.train_batch_size = train_batch_size
        self.val_and_test_batch_size = val_and_test_batch_size
        self.max_len = max_len
        self.val_size = val_size
        self.test_size = test_size
        self.num_workers = num_workers
        self.seed = seed
        self.tokenizer_name = tokenizer_name

        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.dataset: DatasetDict | None = None

    def _tokenize_texts(self, batch):
        enc = self.tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=self.max_len
        )
        enc["labels"] = batch["labels"]
        return enc

    def setup(self, stage: str):
        if self.dataset is not None:
            return

        # Read data
        data = dvc.api.read(self.data_file, mode="r")
        df = pd.read_csv(StringIO(data))

        # Split data on train and test sets
        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            stratify=df["labels"],
            random_state=self.seed,
        )
        # Split train data on train and validation sets
        train_df, val_df = train_test_split(
            df,
            test_size=self.val_size,
            stratify=df["labels"],
            random_state=self.seed,
        )

        train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
        val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))
        test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))
        self.dataset = DatasetDict(train=train_ds, validation=val_ds, test=test_ds)

        self.dataset = self.dataset.map(self._tokenize_texts, batched=True, remove_columns=["text"])
        self.dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset["train"],
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dataset["validation"],
            batch_size=self.val_and_test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.dataset["test"],
            batch_size=self.val_and_test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
