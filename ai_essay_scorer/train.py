import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: DictConfig):
    # fix random seed
    pl.seed_everything(42, workers=True)

    # init logger
    logger = instantiate(cfg.logger)

    # init pl datamodule
    datamodule = instantiate(cfg.datamodule)

    # init pl module with model
    model = instantiate(cfg.pl_module)

    # init pl callbacks
    early_stopping_params = OmegaConf.to_container(cfg["early_stopping_callback"])
    model_checkpoint_params = OmegaConf.to_container(cfg["model_checkpoint_callback"])
    callbacks = [
        pl.callbacks.EarlyStopping(**early_stopping_params),
        pl.callbacks.ModelCheckpoint(**model_checkpoint_params),
    ]

    # init pl trainer
    trainer_params = OmegaConf.to_container(cfg["trainer"])
    trainer = pl.Trainer(callbacks=callbacks, logger=logger, **trainer_params)

    # fit & test model
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
