"""
Visualization functions for SMB nested sequence analysis.

Generates publication-quality figures testing predictions from:
- Xie et al. (2022) - Subspace geometry
- El-Gaby et al. (2024) - Goal-progress cells
- Jensen et al. (2025) - Spacetime attractors
- Whittington et al. (2024) - Activity slots
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional
import json


def set_style():
    """Set publication-quality plotting style."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'savefig.facecolor': 'white',
        'savefig.dpi': 300,
        'figure.dpi': 100,
    })


def plot_training_curves(history: Dict, save_path: Optional[str] = None) -> plt.Figure:
    """Plot training loss and accuracy curves."""
    set_style()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Loss
    ax = axes[0]
    ax.plot(history['train_loss'], alpha=0.7, linewidth=0.5)
    # Smoothed version
    window = 50
    if len(history['train_loss']) > window:
        smoothed = np.convolve(history['train_loss'], np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(history['train_loss'])), smoothed, 'C0', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.set_yscale('log')

    # Accuracy by sequence type
    ax = axes[1]
    colors = {'nested': '#1f77b4', 'flat_repeat': '#ff7f0e', 'unique': '#2ca02c'}
    markers = {'nested': 'o', 'flat_repeat': 's', 'unique': '^'}

    epochs = history['epochs']
    for seq_type, metrics_list in history['val_metrics'].items():
        if metrics_list:
            accs = [m['item_accuracy'] for m in metrics_list]
            ax.plot(epochs[:len(accs)], accs, marker=markers.get(seq_type, 'o'),
                   color=colors.get(seq_type, 'gray'), markersize=4,
                   label=seq_type.replace('_', ' ').title())

    ax.axhline(1/16, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Item Accuracy')
    ax.set_title('Validation Accuracy by Sequence Type')
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    return fig


def plot_subspace_geometry(results: Dict, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot subspace geometry results (Xie et al. analysis).

    Shows:
    - Principal angle matrix for nested sequences
    - Comparison of within-chunk vs between-chunk angles
    - Comparison across sequence types
    """
    set_style()

    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 3, width_ratios=[1.2, 1, 1])

    # Panel A: Principal angle matrix for nested
    ax1 = fig.add_subplot(gs[0])
    nested = results['subspace_geometry']['nested']
    principal_angles = nested.get('principal_angles', {})

    angle_matrix = np.zeros((8, 8))
    angle_matrix[:] = np.nan

    for key, angles in principal_angles.items():
        if isinstance(key, str):
            try:
                i, j = eval(key)
            except:
                continue
        else:
            i, j = key
        mean_angle = np.mean(angles)
        angle_matrix[i, j] = mean_angle
        angle_matrix[j, i] = mean_angle

    np.fill_diagonal(angle_matrix, 0)

    im = ax1.imshow(angle_matrix, cmap='viridis', vmin=0, vmax=90)

    # Add text annotations
    for i in range(8):
        for j in range(8):
            if not np.isnan(angle_matrix[i, j]):
                color = 'white' if angle_matrix[i, j] > 45 else 'black'
                ax1.text(j, i, f'{angle_matrix[i, j]:.0f}',
                        ha='center', va='center', fontsize=7, color=color)

    # Add chunk boundaries
    ax1.axhline(3.5, color='red', linewidth=2, linestyle='--')
    ax1.axvline(3.5, color='red', linewidth=2, linestyle='--')

    ax1.set_xticks(range(8))
    ax1.set_yticks(range(8))
    ax1.set_xticklabels([f'P{i}' for i in range(8)])
    ax1.set_yticklabels([f'P{i}' for i in range(8)])
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Position')
    ax1.set_title('A. Principal Angles (Nested)\nRed lines = chunk boundary')

    plt.colorbar(im, ax=ax1, label='Angle (°)', shrink=0.8)

    # Panel B: Within vs between chunk comparison
    ax2 = fig.add_subplot(gs[1])

    categories = ['Within\nChunk', 'Between\nChunk']
    values = [
        nested.get('mean_within_chunk_angle', 0),
        nested.get('mean_between_chunk_angle', 0),
    ]
    colors = ['#1f77b4', '#d62728']

    bars = ax2.bar(categories, values, color=colors, width=0.6)
    ax2.axhline(90, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Principal Angle (°)')
    ax2.set_title('B. Chunk Structure Effect')
    ax2.set_ylim([0, 100])

    # Add significance bracket if difference is substantial
    diff = values[1] - values[0]
    ax2.annotate(f'Δ = {diff:.1f}°', xy=(0.5, max(values) + 5),
                ha='center', fontsize=10, fontweight='bold')

    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}°', ha='center', va='bottom', fontsize=9)

    # Panel C: Comparison across sequence types
    ax3 = fig.add_subplot(gs[2])

    seq_types = ['nested', 'flat_repeat', 'unique']
    labels = ['Nested', 'Flat Repeat', 'Unique']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    mean_angles = []
    for st in seq_types:
        sg = results['subspace_geometry'].get(st, {})
        mean_angles.append(sg.get('mean_principal_angle', 0))

    bars = ax3.bar(labels, mean_angles, color=colors, width=0.6)
    ax3.axhline(90, color='gray', linestyle='--', alpha=0.5, label='Orthogonal')
    ax3.set_ylabel('Mean Principal Angle (°)')
    ax3.set_title('C. Overall Subspace Separation')
    ax3.set_ylim([0, 100])

    for bar, val in zip(bars, mean_angles):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}°', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    return fig


def plot_future_coding(results: Dict, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot future coding analysis (Jensen et al. spacetime attractor).

    Shows:
    - Current vs next vs future+2 item decoding
    - Chunk boundary effect on future coding
    - Conveyor belt analysis
    """
    set_style()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fc = results['future_coding']

    # Panel A: Decoding by future distance
    ax = axes[0]
    positions = range(len(fc['decoding_by_position']['current_decoding']))

    ax.plot(positions, fc['decoding_by_position']['current_decoding'],
           'o-', label='Current', color='#1f77b4', markersize=6)
    ax.plot(positions, fc['decoding_by_position']['next_decoding'],
           's-', label='Next', color='#ff7f0e', markersize=6)
    ax.plot(positions, fc['decoding_by_position']['future2_decoding'],
           '^-', label='Future+2', color='#2ca02c', markersize=6)

    ax.axhline(1/16, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.axvline(3.5, color='red', linestyle='--', alpha=0.5)  # Chunk boundary

    ax.set_xlabel('Position')
    ax.set_ylabel('Decoding Accuracy')
    ax.set_title('A. Future Coding by Position')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_ylim([0, 1.05])

    # Panel B: Chunk boundary effect
    ax = axes[1]
    boundary = fc['chunk_boundary_effect']

    positions = ['Pre-\nboundary\n(2→3)', 'At\nboundary\n(3→4)', 'Post-\nboundary\n(4→5)']
    values = [
        boundary.get('pre_boundary_accuracy', 0) or 0,
        boundary.get('at_boundary_accuracy', 0) or 0,
        boundary.get('post_boundary_accuracy', 0) or 0,
    ]
    colors = ['#2ca02c', '#d62728', '#2ca02c']

    bars = ax.bar(positions, values, color=colors, width=0.6)
    ax.axhline(1/16, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Next Item Decoding Accuracy')
    ax.set_title('B. Chunk Boundary Effect')
    ax.set_ylim([0, 1.05])

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.0%}', ha='center', va='bottom', fontsize=9)

    # Panel C: Summary statistics
    ax = axes[2]

    categories = ['Current\nItem', 'Next\nItem', 'Future+2\nItem', 'Conveyor\nBelt']
    values = [
        fc['mean_current_decoding'],
        fc['mean_next_decoding'],
        fc['mean_future2_decoding'],
        fc['mean_conveyor_belt'],
    ]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']

    bars = ax.bar(categories, values, color=colors, width=0.6)
    ax.axhline(1/16, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.set_ylabel('Decoding Accuracy')
    ax.set_title('C. Future Coding Summary')
    ax.set_ylim([0, 1.05])

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.0%}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    return fig


def plot_slot_dynamics(results: Dict, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot slot dynamics analysis (Whittington et al.).

    Shows:
    - Position generalization across items
    - Content-structure separation
    - Cross-sequence transfer matrix
    """
    set_style()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel A: Position generalization
    ax = axes[0]
    sd = results['slot_dynamics']['nested']

    categories = ['Standard\nCV', 'Leave-Item\nOut']
    values = [sd['position_cv_accuracy'], sd['position_leave_item_out_accuracy']]
    colors = ['#1f77b4', '#d62728']

    bars = ax.bar(categories, values, color=colors, width=0.5)
    ax.axhline(1/8, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.set_ylabel('Position Decoding Accuracy')
    ax.set_title('A. Position Generalization Test\n(Reusable Slots?)')
    ax.set_ylim([0, 0.5])

    gap = sd['position_generalization_gap']
    ax.annotate(f'Gap = {gap:.1%}', xy=(0.5, max(values) * 1.1),
               ha='center', fontsize=10, fontweight='bold',
               color='red' if gap > 0.05 else 'green')

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.1%}', ha='center', va='bottom', fontsize=9)

    # Panel B: Variance partitioning
    ax = axes[1]

    categories = ['Position\nOnly', 'Item\nOnly', 'Combined', 'Interaction']
    values = [
        sd['mean_position_r2'] * 100,
        sd['mean_item_r2'] * 100,
        sd['mean_combined_r2'] * 100,
        sd['interaction_variance'] * 100,
    ]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']

    bars = ax.bar(categories, values, color=colors, width=0.6)
    ax.set_ylabel('Variance Explained (%)')
    ax.set_title('B. Content-Structure Separation')

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, max(0, bar.get_height()) + 0.5,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    # Panel C: Cross-sequence transfer matrix
    ax = axes[2]
    transfer = results['cross_generalization']['transfer_matrix']

    seq_types = ['nested', 'flat_repeat', 'unique']
    labels = ['Nested', 'Flat', 'Unique']

    matrix = np.zeros((3, 3))
    for i, train_type in enumerate(seq_types):
        for j, test_type in enumerate(seq_types):
            matrix[i, j] = transfer[train_type][test_type]

    im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=0.3)

    for i in range(3):
        for j in range(3):
            color = 'white' if matrix[i, j] > 0.15 else 'black'
            ax.text(j, i, f'{matrix[i, j]:.0%}', ha='center', va='center',
                   fontsize=10, color=color)

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Test Sequence Type')
    ax.set_ylabel('Train Sequence Type')
    ax.set_title('C. Cross-Sequence Transfer')

    plt.colorbar(im, ax=ax, label='Position Accuracy', shrink=0.8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    return fig


def plot_goal_progress(results: Dict, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot goal-progress tiling analysis (El-Gaby et al.).

    Shows:
    - Preferred position distribution
    - Selectivity distribution
    - Chunk modulation
    """
    set_style()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    gp = results['goal_progress']

    # Panel A: Preferred position distribution
    ax = axes[0]
    positions = range(8)
    counts = gp['preferred_position_distribution']

    colors = ['#1f77b4'] * 4 + ['#ff7f0e'] * 4  # Different colors for chunks
    ax.bar(positions, counts, color=colors, width=0.8)

    ax.axhline(np.mean(counts), color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Preferred Position')
    ax.set_ylabel('Number of Neurons')
    ax.set_title('A. Position Preference Distribution\n(Goal-Progress Tiling)')
    ax.set_xticks(positions)

    # Add chunk labels
    ax.text(1.5, max(counts) * 0.95, 'Chunk 1', ha='center', fontsize=9, color='#1f77b4')
    ax.text(5.5, max(counts) * 0.95, 'Chunk 2', ha='center', fontsize=9, color='#ff7f0e')

    # Panel B: Selectivity and smoothness
    ax = axes[1]

    categories = ['Mean\nSelectivity', 'Tuning\nSmoothness', 'Coverage\nUniformity']
    values = [
        gp['mean_selectivity'],
        gp['mean_tuning_smoothness'],
        gp['coverage_uniformity'],
    ]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    bars = ax.bar(categories, values, color=colors, width=0.6)
    ax.set_ylabel('Score')
    ax.set_title('B. Tuning Properties')

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    # Panel C: Chunk modulation
    ax = axes[2]

    categories = ['Within\nChunk', 'Between\nChunk']
    values = [
        gp['mean_within_chunk_correlation'],
        gp['mean_between_chunk_correlation'],
    ]
    colors = ['#1f77b4', '#d62728']

    bars = ax.bar(categories, values, color=colors, width=0.5)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_ylabel('Mean Correlation')
    ax.set_title('C. Chunk Modulation')

    modulation = gp['chunk_modulation']
    ax.annotate(f'Modulation = {modulation:.3f}', xy=(0.5, max(values) * 1.15),
               ha='center', fontsize=10, fontweight='bold',
               color='green' if modulation > 0 else 'red')

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    return fig


def plot_summary_figure(results: Dict, save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a summary figure with key findings.
    """
    set_style()

    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, hspace=0.3, wspace=0.3)

    # Panel A: Subspace geometry summary
    ax = fig.add_subplot(gs[0, 0])
    nested = results['subspace_geometry']['nested']

    metrics = ['Within-Chunk\nAngle', 'Between-Chunk\nAngle', 'Same Local\nRank', 'Diff Local\nRank']
    values = [
        nested.get('mean_within_chunk_angle', 0),
        nested.get('mean_between_chunk_angle', 0),
        nested.get('mean_same_local_rank_angle', 0),
        nested.get('mean_diff_local_rank_angle', 0),
    ]

    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
    bars = ax.bar(metrics, values, color=colors, width=0.6)
    ax.axhline(90, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Principal Angle (°)')
    ax.set_title('A. Hierarchical Structure in Representations')
    ax.set_ylim([0, 100])

    for bar, val in zip(bars, values):
        if val:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.0f}°', ha='center', va='bottom', fontsize=9)

    # Panel B: Future coding
    ax = fig.add_subplot(gs[0, 1])
    fc = results['future_coding']

    x = range(len(fc['decoding_by_position']['next_decoding']))
    ax.fill_between(x, fc['decoding_by_position']['next_decoding'],
                   alpha=0.3, color='#ff7f0e')
    ax.plot(x, fc['decoding_by_position']['next_decoding'],
           'o-', color='#ff7f0e', markersize=6, label='Next Item')

    ax.axvline(3.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.axhline(1/16, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('Position')
    ax.set_ylabel('Next Item Decoding')
    ax.set_title('B. Future Coding Blocked at Chunk Boundary')
    ax.set_ylim([0, 1.1])
    ax.annotate('Chunk\nBoundary', xy=(3.5, 0.9), ha='center', fontsize=9, color='red')

    # Panel C: Slot analysis
    ax = fig.add_subplot(gs[1, 0])
    sd = results['slot_dynamics']['nested']

    # Position vs item subspace angle
    angle = sd['mean_position_item_angle']
    orthogonality = sd['subspace_orthogonality']

    # Create a simple angle visualization
    theta = np.linspace(0, np.pi/2, 100)
    r = 1
    ax.plot(r * np.cos(theta), r * np.sin(theta), 'k-', linewidth=0.5)
    ax.plot([0, r], [0, 0], 'b-', linewidth=2, label='Position subspace')

    actual_angle = np.radians(angle)
    ax.plot([0, r * np.cos(actual_angle)], [0, r * np.sin(actual_angle)],
           'r-', linewidth=2, label='Item subspace')

    ax.annotate(f'{angle:.0f}°', xy=(0.5, 0.3), fontsize=14, fontweight='bold')
    ax.set_xlim([-0.1, 1.2])
    ax.set_ylim([-0.1, 1.2])
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_title(f'C. Position-Item Subspace Angle\n(Orthogonality: {orthogonality:.0%})')
    ax.axis('off')

    # Panel D: Key findings summary
    ax = fig.add_subplot(gs[1, 1])
    ax.axis('off')

    findings = [
        f"1. Chunk Structure: Within-chunk angle ({nested.get('mean_within_chunk_angle', 0):.0f}°) < "
        f"Between-chunk ({nested.get('mean_between_chunk_angle', 0):.0f}°)",
        f"2. Future Coding: {fc['mean_next_decoding']:.0%} next-item decoding, "
        f"blocked at chunk boundaries",
        f"3. Position Generalization: {sd['position_generalization_gap']:.0%} gap "
        f"(NO reusable slots)",
        f"4. Chunk Modulation: {results['goal_progress']['chunk_modulation']:.3f} "
        f"(neurons prefer within-chunk)",
    ]

    ax.text(0.1, 0.9, 'D. Key Findings', fontsize=12, fontweight='bold',
           transform=ax.transAxes)

    for i, finding in enumerate(findings):
        ax.text(0.1, 0.75 - i*0.18, finding, fontsize=10,
               transform=ax.transAxes, wrap=True)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    return fig


def generate_all_figures(results_dir: str, figures_dir: str):
    """Generate all figures for the paper."""
    results_path = Path(results_dir)
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)

    # Load history
    history_file = results_path / 'history.json'
    if history_file.exists():
        with open(history_file) as f:
            history = json.load(f)
        plot_training_curves(history, figures_path / 'fig1_training.png')
        print("Generated fig1_training.png")

    # Load analysis results
    analysis_file = results_path / 'comprehensive_analysis.json'
    if analysis_file.exists():
        with open(analysis_file) as f:
            results = json.load(f)

        # Generate all figures
        plot_subspace_geometry(results, figures_path / 'fig2_subspace_geometry.png')
        print("Generated fig2_subspace_geometry.png")

        plot_future_coding(results, figures_path / 'fig3_future_coding.png')
        print("Generated fig3_future_coding.png")

        plot_slot_dynamics(results, figures_path / 'fig4_slot_dynamics.png')
        print("Generated fig4_slot_dynamics.png")

        plot_goal_progress(results, figures_path / 'fig5_goal_progress.png')
        print("Generated fig5_goal_progress.png")

        plot_summary_figure(results, figures_path / 'fig6_summary.png')
        print("Generated fig6_summary.png")

    print(f"\nAll figures saved to {figures_path}")


if __name__ == "__main__":
    import sys

    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results/main_exp"
    figures_dir = sys.argv[2] if len(sys.argv) > 2 else "figures"

    generate_all_figures(results_dir, figures_dir)
