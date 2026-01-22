#!/usr/bin/env python
"""
Main experiment script for comprehensive SMB study.

Trains a larger GRU network on nested sequences and runs
the full analysis suite to test predictions from:
- Xie et al. (2022) - Subspace geometry
- El-Gaby et al. (2024) - Goal-progress cells
- Jensen et al. (2025) - Spacetime attractors
- Whittington et al. (2024) - Activity slots
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

from task import NestedSequenceTask, SequenceType
from model import create_model
from train import train_model, load_model, evaluate
from analysis import run_comprehensive_analysis


def get_device(device_str: str) -> str:
    """Get appropriate device."""
    if device_str == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    return device_str


def main():
    parser = argparse.ArgumentParser(description='SMB nested sequence experiment')

    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension (default: 256)')
    parser.add_argument('--model_type', type=str, default='gru',
                        choices=['gru', 'lstm'])
    parser.add_argument('--n_layers', type=int, default=1)

    # Task parameters
    parser.add_argument('--n_stimuli', type=int, default=16)

    # Training parameters
    parser.add_argument('--n_epochs', type=int, default=5000,
                        help='Number of training epochs (default: 5000)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.003)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--eval_every', type=int, default=100)

    # Analysis parameters
    parser.add_argument('--n_analysis_trials', type=int, default=500,
                        help='Trials per condition for analysis')

    # Output
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')

    # Mode
    parser.add_argument('--analyze_only', type=str, default=None,
                        help='Path to existing model to analyze')
    parser.add_argument('--skip_training', action='store_true')

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Create save directory
    if args.save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.save_dir = f'results/smb_exp_{timestamp}'
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config['device'] = device
    with open(save_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Saving results to: {save_path}")

    # Create task
    task = NestedSequenceTask(n_stimuli=args.n_stimuli)

    print(f"\n{'='*60}")
    print("TASK CONFIGURATION")
    print(f"{'='*60}")
    print(f"  n_stimuli: {task.n_stimuli}")
    print(f"  seq_length: {task.seq_length}")
    print(f"  trial_length: {task.trial_length}")
    print(f"  input_dim: {task.input_dim}")
    print(f"  output_dim: {task.output_dim}")

    # Training or load
    if args.analyze_only is None and not args.skip_training:
        # Create model
        model = create_model(
            args.model_type,
            input_dim=task.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=task.output_dim,
            n_layers=args.n_layers,
        )

        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n{'='*60}")
        print("MODEL CONFIGURATION")
        print(f"{'='*60}")
        print(f"  Model type: {args.model_type.upper()}")
        print(f"  Hidden dim: {args.hidden_dim}")
        print(f"  Layers: {args.n_layers}")
        print(f"  Parameters: {n_params:,}")

        # Train
        print(f"\n{'='*60}")
        print(f"TRAINING ({args.n_epochs} epochs)")
        print(f"{'='*60}")

        history = train_model(
            model,
            task,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            eval_every=args.eval_every,
            save_dir=str(save_path),
            device=device,
            verbose=True,
        )

        print(f"\n{'='*60}")
        print("FINAL VALIDATION ACCURACY")
        print(f"{'='*60}")
        for seq_type in SequenceType:
            if history['val_metrics'][seq_type.value]:
                item_acc = history['val_metrics'][seq_type.value][-1]['item_accuracy']
                seq_acc = history['val_metrics'][seq_type.value][-1]['sequence_accuracy']
                print(f"  {seq_type.value:12s}: item={item_acc:.1%}, seq={seq_acc:.1%}")

    else:
        load_path = args.analyze_only if args.analyze_only else str(save_path / 'final_model.pt')
        print(f"\nLoading model from {load_path}")
        model, checkpoint = load_model(load_path, task, args.model_type, args.hidden_dim, device)
        history = checkpoint.get('history', None)

    # Run comprehensive analysis
    print(f"\n{'='*60}")
    print("COMPREHENSIVE ANALYSIS")
    print(f"{'='*60}")

    model = model.to(device)
    model.eval()

    analysis_results = run_comprehensive_analysis(
        model, task, n_trials=args.n_analysis_trials, device=device
    )

    # Save analysis results
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int32, np.int64, np.int_)):
            return int(obj)
        if isinstance(obj, (np.float32, np.float64, np.float_)):
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        if isinstance(obj, tuple):
            return str(obj)
        return obj

    with open(save_path / 'comprehensive_analysis.json', 'w') as f:
        json.dump(convert(analysis_results), f, indent=2)

    # Print detailed summary
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")

    # Subspace geometry (Xie et al.)
    print("\n1. SUBSPACE GEOMETRY (Xie et al. 2022)")
    print("-" * 40)
    for seq_type in SequenceType:
        sg = analysis_results['subspace_geometry'][seq_type.value]
        print(f"  {seq_type.value}:")
        if sg.get('mean_within_chunk_angle') is not None:
            print(f"    Within-chunk angle: {sg['mean_within_chunk_angle']:.1f}°")
            print(f"    Between-chunk angle: {sg['mean_between_chunk_angle']:.1f}°")
            print(f"    Chunk difference: {sg['chunk_angle_difference']:.1f}°")

    # Future coding (Jensen et al.)
    print("\n2. FUTURE CODING (Jensen et al. 2025)")
    print("-" * 40)
    fc = analysis_results['future_coding']
    print(f"  Current item decoding: {fc['mean_current_decoding']:.1%}")
    print(f"  Next item decoding: {fc['mean_next_decoding']:.1%}")
    print(f"  Future+2 decoding: {fc['mean_future2_decoding']:.1%}")
    print(f"  Conveyor belt transfer: {fc['mean_conveyor_belt']:.1%}")
    if fc['chunk_boundary_effect'].get('boundary_drop'):
        print(f"  Chunk boundary drop: {fc['chunk_boundary_effect']['boundary_drop']:.1%}")

    # Goal-progress tiling (El-Gaby et al.)
    print("\n3. GOAL-PROGRESS TILING (El-Gaby et al. 2024)")
    print("-" * 40)
    gp = analysis_results['goal_progress']
    print(f"  Mean selectivity: {gp['mean_selectivity']:.2f}")
    print(f"  Highly selective neurons: {gp['n_highly_selective']} ({gp['frac_highly_selective']:.1%})")
    print(f"  Tuning smoothness: {gp['mean_tuning_smoothness']:.2f}")
    print(f"  Chunk modulation: {gp['chunk_modulation']:.3f}")

    # Slot dynamics (Whittington et al.)
    print("\n4. SLOT DYNAMICS (Whittington et al. 2024)")
    print("-" * 40)
    sd = analysis_results['slot_dynamics']['nested']
    print(f"  Position CV accuracy: {sd['position_cv_accuracy']:.1%}")
    print(f"  Leave-item-out accuracy: {sd['position_leave_item_out_accuracy']:.1%}")
    print(f"  Generalization gap: {sd['position_generalization_gap']:.1%}")
    print(f"  Position-item subspace angle: {sd['mean_position_item_angle']:.1f}°")

    # Cross-sequence generalization
    print("\n5. CROSS-SEQUENCE GENERALIZATION")
    print("-" * 40)
    cg = analysis_results['cross_generalization']
    print(f"  Nested→Flat transfer: {cg['transfer_matrix']['nested']['flat_repeat']:.1%}")
    print(f"  Flat→Nested transfer: {cg['transfer_matrix']['flat_repeat']['nested']:.1%}")
    print(f"  Structure specificity: {cg['structure_specificity']:.1%}")

    # Connectivity
    print("\n6. CONNECTIVITY STRUCTURE")
    print("-" * 40)
    conn = analysis_results['connectivity']
    print(f"  Spectral radius: {conn['spectral_radius']:.2f}")
    print(f"  Effective rank: {conn['effective_rank']:.1f}")

    # Attractor dynamics
    print("\n7. ATTRACTOR DYNAMICS")
    print("-" * 40)
    for seq_type in ['nested', 'flat_repeat', 'unique']:
        attr = analysis_results['attractor_dynamics'][seq_type]
        print(f"  {seq_type}:")
        print(f"    Temporal variance: {attr['temporal_variance']:.4f}")
        print(f"    Trajectory straightness: {attr['trajectory_straightness']:.2f}")

    print(f"\n{'='*60}")
    print(f"Results saved to: {save_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
