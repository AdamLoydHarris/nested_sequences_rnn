"""
Training utilities for sequence working memory models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from tqdm import tqdm

from task import NestedSequenceTask, SequenceType, TrialInfo
from model import create_model


def compute_accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[float, float]:
    """
    Compute item-level and sequence-level accuracy.

    Args:
        outputs: (batch, seq_len, n_classes) logits
        targets: (batch, seq_len) target classes
        mask: (batch, seq_len) loss mask

    Returns:
        item_accuracy: Fraction of correct items (where mask=1)
        sequence_accuracy: Fraction of completely correct sequences
    """
    predictions = outputs.argmax(dim=-1)  # (batch, seq_len)

    # Item accuracy
    correct_items = (predictions == targets) & (mask > 0)
    total_items = mask.sum()
    item_accuracy = correct_items.sum().float() / total_items if total_items > 0 else 0.0

    # Sequence accuracy
    batch_size = outputs.shape[0]
    seq_correct = []
    for i in range(batch_size):
        trial_mask = mask[i] > 0
        if trial_mask.any():
            all_correct = (predictions[i][trial_mask] == targets[i][trial_mask]).all()
            seq_correct.append(all_correct.float())

    sequence_accuracy = torch.stack(seq_correct).mean() if seq_correct else torch.tensor(0.0)

    return item_accuracy.item(), sequence_accuracy.item()


def evaluate(
    model: nn.Module,
    task: NestedSequenceTask,
    n_trials: int = 200,
    device: str = 'cpu',
) -> Dict[str, Dict]:
    """
    Evaluate model on each sequence type.

    Returns:
        Dict mapping sequence type to metrics dict
    """
    model.eval()
    results = {}

    with torch.no_grad():
        for seq_type in SequenceType:
            inputs, targets, mask, infos = task.generate_batch(n_trials, seq_type)
            inputs = inputs.to(device)
            targets = targets.to(device)
            mask = mask.to(device)

            outputs, _ = model(inputs)
            item_acc, seq_acc = compute_accuracy(outputs, targets, mask)

            results[seq_type.value] = {
                'item_accuracy': item_acc,
                'sequence_accuracy': seq_acc,
            }

    return results


def train_model(
    model: nn.Module,
    task: NestedSequenceTask,
    n_epochs: int = 5000,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    grad_clip: float = 1.0,
    eval_every: int = 100,
    save_dir: Optional[str] = None,
    device: str = 'cpu',
    verbose: bool = True,
) -> Dict:
    """
    Train model on sequence working memory task.

    Args:
        model: Neural network model
        task: Task instance
        n_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: L2 regularization
        grad_clip: Gradient clipping norm
        eval_every: Evaluate every N epochs
        save_dir: Directory to save checkpoints
        device: Device to train on
        verbose: Print progress

    Returns:
        Training history dict
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(reduction='none')

    history = {
        'train_loss': [],
        'epochs': [],
        'val_metrics': {st.value: [] for st in SequenceType},
    }

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

    best_loss = float('inf')
    pbar = tqdm(range(n_epochs), disable=not verbose, desc="Training")

    for epoch in pbar:
        model.train()

        # Generate balanced batch
        inputs, targets, mask, infos = task.generate_batch(batch_size, balanced=True)
        inputs = inputs.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        # Forward pass
        outputs, _ = model(inputs)

        # Compute masked loss
        loss_per_step = criterion(outputs.view(-1, task.output_dim), targets.view(-1))
        loss_per_step = loss_per_step.view(batch_size, -1)
        masked_loss = (loss_per_step * mask).sum() / mask.sum()

        # Backward pass
        optimizer.zero_grad()
        masked_loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        history['train_loss'].append(masked_loss.item())

        # Update progress bar
        pbar.set_postfix({'loss': f'{masked_loss.item():.4f}'})

        # Evaluation
        if (epoch + 1) % eval_every == 0:
            val_metrics = evaluate(model, task, n_trials=200, device=device)
            history['epochs'].append(epoch + 1)

            for seq_type in SequenceType:
                history['val_metrics'][seq_type.value].append(val_metrics[seq_type.value])

            if verbose:
                msg = f"Epoch {epoch+1}: "
                for st in SequenceType:
                    acc = val_metrics[st.value]['item_accuracy']
                    msg += f"{st.value[:6]}={acc:.1%} "
                tqdm.write(msg)

            # Save best model
            if masked_loss.item() < best_loss and save_dir:
                best_loss = masked_loss.item()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'history': history,
                }, save_path / 'best_model.pt')

    # Save final model
    if save_dir:
        torch.save({
            'epoch': n_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': history['train_loss'][-1],
            'history': history,
        }, save_path / 'final_model.pt')

        # Save history
        with open(save_path / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    return history


def load_model(
    checkpoint_path: str,
    task: NestedSequenceTask,
    model_type: str = 'gru',
    hidden_dim: int = 256,
    device: str = 'cpu',
) -> Tuple[nn.Module, Dict]:
    """
    Load model from checkpoint.

    Returns:
        model: Loaded model
        checkpoint: Full checkpoint dict
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Detect model type from checkpoint keys
    state_dict = checkpoint['model_state_dict']
    if any('gru' in k for k in state_dict.keys()):
        model_type = 'gru'
    elif any('lstm' in k for k in state_dict.keys()):
        model_type = 'lstm'

    # Detect hidden dim from weights
    for key, value in state_dict.items():
        if 'weight_hh' in key:
            if model_type == 'gru':
                hidden_dim = value.shape[1]
            else:
                hidden_dim = value.shape[1]
            break

    model = create_model(
        model_type,
        input_dim=task.input_dim,
        hidden_dim=hidden_dim,
        output_dim=task.output_dim,
    )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, checkpoint


if __name__ == "__main__":
    # Quick test
    task = NestedSequenceTask()
    model = create_model('gru', task.input_dim, 128, task.output_dim)

    print("Testing training loop...")
    history = train_model(
        model, task,
        n_epochs=100,
        batch_size=32,
        eval_every=50,
        verbose=True,
    )
    print("Training test complete.")
