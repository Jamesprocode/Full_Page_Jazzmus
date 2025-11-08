"""
Training script for full-page jazz leadsheet recognition.

Supports:
1. System-level checkpoint (expect bad results)
2. Training from scratch
3. Training with pretrained weights
4. Curriculum learning (coming soon)
"""

import gc
import torch
import gin
from pathlib import Path

from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import WandbLogger

from jazzmus.dataset.full_page_datamodule import FullPageDataModule
from jazzmus.smt_trainer import SMT_Trainer
from jazzmus.utils.file_utils import check_folders, print_smt_batch


def train_full_page(
    config=None,
    debug=False,
    data_path="data/handwritten",
    output_name="full_page",
    epochs=100,
    batch_size=2,
    accumulate_grad_batches=1,
    lr=1e-4,
    load_pretrained=False,
    freeze_encoder=False,
    patience=10,
    checkpoint=None,
):
    """
    Train SMT model for full-page jazz leadsheet recognition.

    Args:
        config: Path to gin config file
        debug: Debug mode (skip logging, use small dataset)
        data_path: Path to dataset (train/val/test splits)
        output_name: Name for output weights folder
        epochs: Number of training epochs
        batch_size: Batch size
        accumulate_grad_batches: Gradient accumulation steps
        lr: Learning rate
        load_pretrained: Load pretrained system-level weights
        freeze_encoder: Freeze encoder when using pretrained weights
        patience: Early stopping patience
        checkpoint: Path to checkpoint to resume training
    """

    # Setup
    gc.collect()
    torch.cuda.empty_cache()
    seed_everything(seed=42, workers=True)

    # Parse config
    if config:
        gin.parse_config_file(config)

    check_folders()

    # Print configuration
    print("=" * 60)
    print("FULL-PAGE JAZZ LEADSHEET RECOGNITION TRAINING")
    print("=" * 60)
    print(f"  Data path: {data_path}")
    print(f"  Output name: {output_name}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Load pretrained: {load_pretrained}")
    print(f"  Freeze encoder: {freeze_encoder}")
    print(f"  Debug mode: {debug}")
    print("=" * 60)

    # Create datamodule
    print("\nInitializing datamodule...")
    datamodule = FullPageDataModule(
        data_path=data_path,
        vocab_name=f"{output_name}_vocab",
        batch_size=batch_size,
    )
    datamodule.setup(stage="fit")

    # Print batch info
    print("\nSample batch:")
    print_smt_batch(datamodule.train_dataloader())

    # Get dimensions
    max_height = max(
        datamodule.train_set.get_max_hw()[0],
        datamodule.val_set.get_max_hw()[0],
        datamodule.test_set.get_max_hw()[0],
    )

    max_width = max(
        datamodule.train_set.get_max_hw()[1],
        datamodule.val_set.get_max_hw()[1],
        datamodule.test_set.get_max_hw()[1],
    )

    max_len = max(
        datamodule.train_set.get_max_seqlen(),
        datamodule.val_set.get_max_seqlen(),
        datamodule.test_set.get_max_seqlen(),
    )

    print(f"\n✓ Max image height: {max_height}")
    print(f"✓ Max image width: {max_width}")
    print(f"✓ Max sequence length: {max_len}")
    print(f"✓ Vocabulary size: {len(datamodule.train_set.w2i)}")

    # Create model
    print("\nCreating SMT model...")
    model = SMT_Trainer(
        maxh=int(max_height),
        maxw=int(max_width),
        maxlen=int(max_len),
        out_categories=len(datamodule.train_set.w2i),
        padding_token=datamodule.train_set.padding_token,
        in_channels=1,
        w2i=datamodule.train_set.w2i,
        i2w=datamodule.train_set.i2w,
        texture="jazz",
        fold=0,
        lr=lr,
        load_pretrained=load_pretrained,
    )

    if freeze_encoder and load_pretrained:
        print("\n⚠ Freezing encoder layers (pretrained feature extraction)")
        for name, param in model.model.encoder.named_parameters():
            param.requires_grad = False
        print("✓ Encoder frozen")

    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor="val/loss",
            patience=patience,
            mode="min",
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=f"weights/{output_name}",
            filename="{epoch}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Setup logger
    logger = None
    if not debug:
        logger = WandbLogger(
            project="jazz-leadsheet-ocr",
            name=f"full_page_{output_name}",
            tags=["full-page", "smt", "jazz"],
        )

    # Create trainer
    print("\nCreating trainer...")
    trainer = Trainer(
        max_epochs=epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
        devices=[0],  # Use first GPU
        accelerator="auto",
        precision="16-mixed",
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=10,
        num_sanity_val_steps=2 if debug else 0,
        limit_train_batches=10 if debug else 1.0,
        limit_val_batches=5 if debug else 1.0,
    )

    # Train
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    if checkpoint:
        print(f"\nResuming from checkpoint: {checkpoint}")
        trainer.fit(
            model,
            datamodule=datamodule,
            ckpt_path=checkpoint,
        )
    else:
        trainer.fit(
            model,
            datamodule=datamodule,
        )

    # Test
    print("\n" + "=" * 60)
    print("RUNNING EVALUATION")
    print("=" * 60)
    trainer.test(datamodule=datamodule)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"✓ Weights saved to: weights/{output_name}/")


if __name__ == "__main__":
    import fire

    fire.Fire(train_full_page)
