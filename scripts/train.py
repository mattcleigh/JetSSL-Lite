"""Basic training script."""

import logging
from pathlib import Path

import h5py
import hydra
import lightning.pytorch as pl
import rootutils
import torch as T
from omegaconf import DictConfig

root = rootutils.setup_root(search_from=__file__, pythonpath=True)
cfg_path = str(root / "configs")

from mltools.mltools.hydra_utils import (
    instantiate_collection,
    log_hyperparameters,
    print_config,
    reload_original_config,
    save_config,
)
from mltools.mltools.torch_utils import to_numpy
from mltools.mltools.utils import save_declaration

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=cfg_path, config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    """Main training script."""
    log.info("Setting up full job config")

    if cfg.full_resume:
        log.info("Attempting to resume previous job")
        old_cfg = reload_original_config(ckpt_flag=cfg.ckpt_flag)
        if old_cfg is not None:
            cfg = old_cfg
    print_config(cfg)

    log.info(f"Setting seed to: {cfg.seed}")
    pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Setting matrix precision to: {cfg.precision}")
    T.set_float32_matmul_precision(cfg.precision)

    log.info("Instantiating the data module")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    log.info("Instantiating the model")
    if cfg.weight_ckpt_path:
        log.info(f"Loading model weights from checkpoint: {cfg.ckpt_path}")
        model_class = hydra.utils.get_class(cfg.model._target_)
        model = model_class.load_from_checkpoint(cfg.ckpt_path, map_location="cpu")
    else:
        model = hydra.utils.instantiate(
            cfg.model,
            data_sample=datamodule.get_data_sample(),
            n_classes=datamodule.get_n_classes(),
        )

    if cfg.compile:
        log.info(f"Compiling the model using torch 2.0: {cfg.compile}")
        model = T.compile(model, mode=cfg.compile)

    log.info("Instantiating all callbacks")
    callbacks = instantiate_collection(cfg.callbacks)

    log.info("Instantiating the logger")
    logger = hydra.utils.instantiate(cfg.logger)

    log.info("Instantiating the trainer")
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    log.info("Logging all hyperparameters")
    log_hyperparameters(cfg, model, trainer)
    log.info(model)

    log.info("Saving config so job can be resumed")
    save_config(cfg)

    log.info("Starting training!")
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    log.info("Checking if training finished correctly")
    if trainer.state.status == "finished":
        log.info(" -- YES!! -- ")
        save_declaration()

    if cfg.save_test_preds:
        log.info("Running inference on test set")

        log.info("Attempting to load best checkpoint")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if not ckpt_path:
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None

        log.info("Running inference on test set")
        outputs = trainer.predict(
            model=model, datamodule=datamodule, ckpt_path=ckpt_path
        )

        log.info("Combining predictions across dataset")
        keys = list(outputs[0].keys())
        score_dict = {k: T.vstack([o[k] for o in outputs]) for k in keys}

        log.info("Saving outputs")
        output_dir = Path(cfg.full_path, "outputs")
        print(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_dir / "test_set.h5", mode="w") as file:
            for k in keys:
                file.create_dataset(k, data=to_numpy(score_dict[k]))


if __name__ == "__main__":
    main()
