"""Training loop and checkpoints for the four models we train from scratch."""
from __future__ import annotations

import json
import math
import time
from functools import partial
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from ..config import Config, set_seed
from ..datasets.noise import PairDataset
from ..datasets.vocab import PAD, Vocab, collate
from ..datasets.word_data import DataBundle
from ..evaluation.metrics import evaluate
from ..models.decode import hypothesis_texts, predict
from ..models.gru import GRUSeq2Seq
from ..models.transformer import CharTransformer

# Hyper-parameters of each architecture
HP: dict[str, dict[str, Any]] = {
    "gru": {
        "emb": 192,
        "hid": 256,
        "layers": 2,
        "drop": 0.25,
        "lr": 1e-3,
        "label_smoothing": 0.0,
    },
    "transformer": {
        "d": 256,
        "heads": 4,
        "encoder_layers": 3,
        "decoder_layers": 3,
        "ff": 1024,
        "drop": 0.2,
        "peak_lr": 7e-4,
        "warmup": 1000,
        "label_smoothing": 0.1,
    },
}

# Model name -> (architecture, uses noise: 0.0 = no, 0.5 = yes)
EXPERIMENTS: dict[str, tuple[str, float]] = {
    "GRU-clean": ("gru", 0.0),
    "GRU-noisy": ("gru", 0.5),
    "Transformer-clean": ("transformer", 0.0),
    "Transformer-noisy": ("transformer", 0.5),
}


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def make_model(
    architecture: str,
    source_vocab: Vocab,
    target_vocab: Vocab,
    hyperparameters: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    """Create the model for an architecture name ('gru' or 'transformer')."""
    # Build a GRU model
    if architecture == "gru":
        model = GRUSeq2Seq(
            len(source_vocab),
            len(target_vocab),
            hyperparameters["emb"],
            hyperparameters["hid"],
            hyperparameters["layers"],
            hyperparameters["drop"],
        )
    # Otherwise build the Transformer
    else:
        model = CharTransformer(
            len(source_vocab),
            len(target_vocab),
            hyperparameters["d"],
            hyperparameters["heads"],
            hyperparameters["encoder_layers"],
            hyperparameters["decoder_layers"],
            hyperparameters["ff"],
            hyperparameters["drop"],
        )
    # Move the model to the CPU or GPU
    return model.to(device)


def load_model(
    name: str,
    config: Config,
    source_vocab: Vocab,
    target_vocab: Vocab,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a saved model from its checkpoint file."""
    # Checkpoint file of this model
    checkpoint_path = config.out_dir / f"{name}.pt"
    # Read the checkpoint
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Which architecture was saved
    architecture = payload["arch"]
    # Rebuild the model with the saved settings
    model = make_model(architecture, source_vocab, target_vocab, payload["hp"], device)
    # Load the trained weights
    model.load_state_dict(payload["state"])
    model.eval()
    return model, payload


@torch.no_grad()
def validation_loss(
    model: nn.Module, loader: DataLoader[Any], device: torch.device
) -> float:
    """Average loss per character on the validation pairs."""
    model.eval()
    # Sum of the losses and number of characters
    total = 0.0
    tokens = 0
    # Go through the validation batches
    for source, target in loader:
        # Move the batch to the device
        source = source.to(device)
        target = target.to(device)
        # Predict each next character (teacher forcing)
        logits = model(source, target[:, :-1])
        # Add the loss of this batch
        total += F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            target[:, 1:].reshape(-1),
            ignore_index=PAD,
            reduction="sum",
        ).item()
        # Count the real (non-padding) characters
        tokens += (target[:, 1:] != PAD).sum().item()
    # Loss per character
    return total / max(1, tokens)


def validation_accuracy(
    model: nn.Module,
    config: Config,
    bundle: DataBundle,
    source_vocab: Vocab,
    target_vocab: Vocab,
    device: torch.device,
) -> float:
    """Top-1 accuracy on the validation words (greedy decoding)."""
    # Decode the validation words (beam 1 = greedy)
    predictions = predict(
        model,
        bundle.val_words,
        source_vocab,
        target_vocab,
        device,
        beam=1,
        batch_size=config.eval_batch_size,
        max_len=config.max_decode_len,
    )
    # Keep only the texts
    texts = hypothesis_texts(predictions)
    # Accepted spellings of each word
    references = [bundle.val_refs[word] for word in bundle.val_words]
    # Top-1 accuracy
    return evaluate(texts, references, ks=(1,))["top1"]


def train_model(
    name: str,
    config: Config,
    bundle: DataBundle,
    source_vocab: Vocab,
    target_vocab: Vocab,
    device: torch.device,
    overwrite: bool = False,
) -> tuple[nn.Module, list[dict[str, Any]]]:
    """Train one model (or load it if saved); returns the model and its history."""
    # Look up the architecture and noise setting
    architecture, default_probability = EXPERIMENTS[name]
    # Clean models get no noise
    noisy_probability = config.noisy_prob if default_probability > 0 else 0.0
    # Settings of this architecture
    hyperparameters = HP[architecture]
    # Where this checkpoint is saved
    checkpoint_path = config.out_dir / f"{name}.pt"

    # Reuse a saved model instead of retraining
    if checkpoint_path.exists() and not overwrite:
        # Read the checkpoint
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = make_model(
            architecture, source_vocab, target_vocab, payload["hp"], device
        )
        # Load the saved weights
        model.load_state_dict(payload["state"])
        model.eval()
        print(f"[{name}] loaded checkpoint ({count_parameters(model):,} params)")
        return model, payload["history"]

    # Same seed for repeatable training
    set_seed(config.seed)
    # Create a fresh model
    model = make_model(
        architecture, source_vocab, target_vocab, hyperparameters, device
    )
    print(
        f"[{name}] {architecture}, {count_parameters(model):,} params, "
        f"noisy_prob={noisy_probability}"
    )

    # Training data (with noise for the noisy models)
    dataset = PairDataset(
        bundle.train_pairs,
        noisy_probability,
        (config.noise_low, config.noise_high),
        seed=config.seed,
        weights=config.noise_weights or None,
    )
    # Training batches, shuffled every epoch
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=partial(collate, source_vocab=source_vocab, target_vocab=target_vocab),
        num_workers=config.num_workers,
        generator=torch.Generator().manual_seed(config.seed),
    )
    # Validation data (always clean)
    validation_dataset = PairDataset(bundle.val_pairs_loss)
    # Validation batches
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=512,
        shuffle=False,
        collate_fn=partial(collate, source_vocab=source_vocab, target_vocab=target_vocab),
    )

    # Loss that ignores padding
    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD, label_smoothing=hyperparameters["label_smoothing"]
    )
    # Optimiser and learning-rate schedule per architecture
    if architecture == "gru":
        # GRU: AdamW
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=hyperparameters["lr"], weight_decay=1e-4
        )
        # Halve the rate when accuracy stalls
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2
        )
        # Schedule is updated once per epoch
        per_step = False
    # Transformer: Adam with warm-up
    else:
        # Transformer: Adam
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=hyperparameters["peak_lr"],
            betas=(0.9, 0.98),
            eps=1e-9,
        )
        # Number of warm-up steps
        warmup = hyperparameters["warmup"]
        # Rate rises during warm-up, then decays
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: min((step + 1) / warmup, math.sqrt(warmup / (step + 1))),
        )
        # Schedule is updated every batch
        per_step = True

    # Mixed precision only on a GPU
    use_amp = config.amp and device.type == "cuda"
    # Stops small gradients from underflowing
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    # One record per epoch
    history: list[dict[str, Any]] = []
    # Best validation accuracy so far
    best_accuracy = -1.0
    # Weights of the best epoch
    best_state: dict[str, torch.Tensor] | None = None
    # Epochs in a row without improvement
    bad_epochs = 0

    # Train epoch by epoch
    for epoch in range(1, config.epochs + 1):
        # New noise every epoch
        dataset.set_epoch(epoch)
        # Training mode (dropout on)
        model.train()
        # Start the epoch timer
        started = time.time()
        # Loss totals of this epoch
        total_loss = 0.0
        token_count = 0
        # Go through the training batches
        for source, target in loader:
            source = source.to(device)
            target = target.to(device)
            # Input is the output shifted one step
            target_in, target_out = target[:, :-1], target[:, 1:]
            # Forward pass (mixed precision if on)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                logits = model(source, target_in)
            # Loss against the true next characters
            loss = criterion(
                logits.reshape(-1, logits.size(-1)).float(), target_out.reshape(-1)
            )
            # Clear the old gradients
            optimizer.zero_grad(set_to_none=True)
            # Backpropagate
            scaler.scale(loss).backward()
            # Undo the scaling before clipping
            scaler.unscale_(optimizer)
            # Clip very large gradients
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update the weights
            scaler.step(optimizer)
            scaler.update()
            # Transformer: update the schedule every batch
            if per_step:
                scheduler.step()
            # Real characters in this batch
            tokens = (target_out != PAD).sum().item()
            # Add to the epoch total
            total_loss += loss.item() * tokens
            token_count += tokens

        # Evaluate on the validation data
        validation_loss_value = validation_loss(model, validation_loader, device)
        validation_accuracy_value = validation_accuracy(
            model, config, bundle, source_vocab, target_vocab, device
        )
        # GRU: update the schedule once per epoch
        if not per_step:
            scheduler.step(validation_accuracy_value)
        # Current learning rate
        learning_rate = optimizer.param_groups[0]["lr"]
        # Record this epoch
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(1, token_count),
                "val_loss": validation_loss_value,
                "val_top1": validation_accuracy_value,
                "lr": learning_rate,
                "seconds": time.time() - started,
            }
        )
        # Used to mark the best epoch in the log
        marker = ""
        # Is this the best epoch so far?
        if validation_accuracy_value > best_accuracy + 1e-4:
            best_accuracy = validation_accuracy_value
            bad_epochs = 0
            # Keep a copy of these weights
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            marker = "  *best*"
        # No improvement this epoch
        else:
            bad_epochs += 1
        print(
            f"[{name}] ep {epoch:02d} train {history[-1]['train_loss']:.3f} "
            f"val_loss {validation_loss_value:.3f} val_top1 {validation_accuracy_value:.3f} "
            f"lr {learning_rate:.2e} {time.time() - started:.0f}s{marker}",
            flush=True,
        )
        # Stop if there was no progress for too long
        if bad_epochs >= config.patience:
            print(f"[{name}] early stopping", flush=True)
            break

    # Go back to the best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    # Everything stored in the checkpoint
    payload = {
        "state": best_state if best_state is not None else model.state_dict(),
        "history": history,
        "arch": architecture,
        "hp": hyperparameters,
        "noisy_prob": noisy_probability,
        "config": config.to_dict(),
        "epoch": len(history),
        "best_acc": best_accuracy,
    }
    # Write the checkpoint file
    torch.save(payload, checkpoint_path)
    return model, history



def train_all(
    config: Config,
    bundle: DataBundle,
    source_vocab: Vocab,
    target_vocab: Vocab,
    device: torch.device,
    overwrite: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Train every configured model, then save the history and curves."""
    # History of every model
    histories: dict[str, list[dict[str, Any]]] = {}
    # Train the models one by one
    for name in config.models:
        # Reject unknown model names
        if name not in EXPERIMENTS:
            raise ValueError(f"Unknown model: {name}")
        _, history = train_model(
            name, config, bundle, source_vocab, target_vocab, device, overwrite
        )
        histories[name] = history
    # Save all the histories
    history_path = config.out_dir / "training_history.json"
    history_path.write_text(
        json.dumps(histories, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    # Plot validation loss and accuracy
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    # One line per model
    for name, history in histories.items():
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["val_loss"] for row in history], label=name)
        axes[1].plot(epochs, [row["val_top1"] for row in history], label=name)
    axes[0].set_title("Validation loss")
    axes[0].set_xlabel("epoch")
    axes[1].set_title("Validation top-1")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    figure.tight_layout()
    # Save the plot
    figure.savefig(config.out_dir / "training_curves.png", dpi=150)
    plt.close(figure)
    return histories
