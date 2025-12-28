import hydra
import pytorch_lightning as pl
from module import LightningTextClassifier
from omegaconf import DictConfig, OmegaConf

from data import LightningDataModule


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(42, workers=True)

    datamodule = LightningDataModule(
        data_file="../data/raw_data.csv",
        train_batch_size=16,
        val_and_test_batch_size=32,
        max_len=512,
        val_size=0.15,
        test_size=0.15,
        num_workers=0,
        seed=42,
        tokenizer_name="bert-base-uncased",
    )

    model = LightningTextClassifier(
        hf_model="bert-base-uncased", lr=3e-4, weight_decay=0.1, freeze_n_layers=1, num_classes=6
    )

    callbacks = [
        pl.callbacks.EarlyStopping(monitor="val_f1_score", patience=3, verbose=True, mode="max"),
        pl.callbacks.ModelCheckpoint(
            monitor="val_f1_score", save_top_k=1, mode="max", verbose=True
        ),
    ]

    trainer = pl.Trainer(
        deterministic=True,
        max_epochs=10,
        accelerator="auto",
        enable_progress_bar=True,
        log_every_n_steps=25,
        callbacks=callbacks,
    )

    trainer.fit(
        model,
        datamodule=datamodule,
    )

    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
