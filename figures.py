#!/usr/bin/env python
"""

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from pathlib import Path
import json
import torch
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

# Set up publication-quality defaults
plt.rcParams.update({
    'font.family': 'Helvetica Neue',
    'font.size': 8,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.titlesize': 11,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'lines.linewidth': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Nature-inspired color palettes
COLORS = {
    # Main palette - sophisticated blues and warm accents
    'primary': '#2E4057',      # Deep blue-gray
    'secondary': '#048A81',    # Teal
    'accent1': '#E85D04',      # Warm orange
    'accent2': '#9D4EDD',      # Purple
    'accent3': '#06D6A0',      # Mint green

    # Chunk colors
    'chunk1': '#3A86FF',       # Bright blue
    'chunk2': '#FF006E',       # Magenta

    # Sequential palette for heatmaps
    'seq_start': '#F0F9E8',
    'seq_mid': '#7BCCC4',
    'seq_end': '#0868AC',

    # Diverging palette
    'div_neg': '#D73027',
    'div_mid': '#FFFFBF',
    'div_pos': '#1A9850',

    # Grayscale
    'gray_light': '#E5E5E5',
    'gray_mid': '#A3A3A3',
    'gray_dark': '#525252',

    # Background
    'bg': '#FAFAFA',
}

# Position colors (gradient from chunk1 to chunk2)
def get_position_colors(n_positions=8):
    """Generate position colors transitioning between chunks."""
    colors = []
    cmap1 = LinearSegmentedColormap.from_list('chunk1', ['#3A86FF', '#8338EC'])
    cmap2 = LinearSegmentedColormap.from_list('chunk2', ['#FF006E', '#FB5607'])

    for i in range(n_positions):
        if i < 4:
            colors.append(cmap1(i / 3))
        else:
            colors.append(cmap2((i - 4) / 3))
    return colors

POS_COLORS = get_position_colors()


def add_panel_label(ax, label, x=-0.15, y=1.1, fontsize=12, fontweight='bold'):
    """Add panel label (a, b, c, etc.) to axis."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
            fontweight=fontweight, va='top', ha='left')


def create_custom_cmap(colors, name='custom'):
    """Create custom colormap from list of colors."""
    return LinearSegmentedColormap.from_list(name, colors)


# =============================================================================
# FIGURE 1: Task and Model Overview
# =============================================================================

def figure1_task_overview(save_path):
    """
    Figure 1: Task structure and theoretical frameworks.

    Panel a: Task trial structure
    Panel b: Sequence types (nested, flat, unique)
    Panel c: Theoretical predictions schematic
    """
    fig = plt.figure(figsize=(7.2, 6))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1.2],
                           hspace=0.4, wspace=0.4)

    # Panel A: Trial structure timeline
    ax_a = fig.add_subplot(gs[0, :])
    add_panel_label(ax_a, 'a')

    # Draw trial timeline
    phases = [
        ('Fixation', 3, COLORS['gray_mid']),
        ('Encoding', 16, COLORS['chunk1']),
        ('Delay', 8, COLORS['gray_light']),
        ('Recall', 16, COLORS['chunk2']),
    ]

    x = 0
    y = 0.5
    for phase, duration, color in phases:
        rect = FancyBboxPatch((x, y - 0.3), duration, 0.6,
                               boxstyle="round,pad=0.02,rounding_size=0.1",
                               facecolor=color, edgecolor='none', alpha=0.8)
        ax_a.add_patch(rect)
        ax_a.text(x + duration/2, y, phase, ha='center', va='center',
                 fontsize=8, fontweight='medium',
                 color='white' if color != COLORS['gray_light'] else COLORS['gray_dark'])
        x += duration + 1

    # Add stimulus markers during encoding
    for i in range(8):
        marker_x = 4 + i * 2
        ax_a.plot(marker_x, 0.9, 's', markersize=6, color=POS_COLORS[i])
        ax_a.text(marker_x, 1.15, f'S{i+1}', ha='center', va='bottom', fontsize=6)

    ax_a.set_xlim(-1, 50)
    ax_a.set_ylim(0, 1.4)
    ax_a.axis('off')
    ax_a.set_title('Trial Structure', fontweight='bold', pad=10)

    # Panel B: Sequence types
    ax_b = fig.add_subplot(gs[1, 0])
    add_panel_label(ax_b, 'b')

    # Draw sequence patterns
    seq_types = [
        ('Nested\n(ABABCDCD)', [0, 1, 0, 1, 2, 3, 2, 3]),
        ('Flat Repeat\n(ABCDABCD)', [0, 1, 2, 3, 0, 1, 2, 3]),
        ('Unique\n(EFGHIJKL)', [0, 1, 2, 3, 4, 5, 6, 7]),
    ]

    item_colors = ['#3A86FF', '#8338EC', '#FF006E', '#FB5607',
                   '#06D6A0', '#FFD60A', '#F72585', '#4CC9F0']

    for row, (name, pattern) in enumerate(seq_types):
        y_pos = 2 - row * 0.8
        ax_b.text(-0.5, y_pos, name, ha='right', va='center', fontsize=7)

        for col, item in enumerate(pattern):
            rect = FancyBboxPatch((col * 0.35, y_pos - 0.15), 0.3, 0.3,
                                   boxstyle="round,pad=0.01,rounding_size=0.05",
                                   facecolor=item_colors[item], edgecolor='white',
                                   linewidth=0.5)
            ax_b.add_patch(rect)

            # Add chunk boundary marker for nested
            if row == 0 and col == 3:
                ax_b.axvline(col * 0.35 + 0.33, y_pos - 0.2, y_pos + 0.2,
                           color=COLORS['gray_dark'], linestyle='--', linewidth=0.8)

    ax_b.set_xlim(-2, 3.5)
    ax_b.set_ylim(-0.5, 2.5)
    ax_b.axis('off')
    ax_b.set_title('Sequence Conditions', fontweight='bold')

    # Panel C: Subspace geometry schematic
    ax_c = fig.add_subplot(gs[1, 1])
    add_panel_label(ax_c, 'c')

    # Draw 2D subspace illustration
    theta1, theta2 = np.radians(20), np.radians(70)

    # Subspace 1 (chunk 1)
    ax_c.annotate('', xy=(np.cos(theta1), np.sin(theta1)), xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color=COLORS['chunk1'], lw=2))
    ax_c.annotate('', xy=(np.cos(theta1 + np.pi/2) * 0.6, np.sin(theta1 + np.pi/2) * 0.6),
                 xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color=COLORS['chunk1'], lw=2, alpha=0.5))

    # Subspace 2 (chunk 2)
    ax_c.annotate('', xy=(np.cos(theta2), np.sin(theta2)), xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color=COLORS['chunk2'], lw=2))
    ax_c.annotate('', xy=(np.cos(theta2 + np.pi/2) * 0.6, np.sin(theta2 + np.pi/2) * 0.6),
                 xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color=COLORS['chunk2'], lw=2, alpha=0.5))

    # Angle arc
    angles = np.linspace(theta1, theta2, 30)
    ax_c.plot(0.4 * np.cos(angles), 0.4 * np.sin(angles),
             color=COLORS['gray_dark'], lw=1, linestyle='-')
    ax_c.text(0.5, 0.35, 'θ', fontsize=10, style='italic')

    ax_c.text(1.05, 0.4, 'Chunk 1\nsubspace', fontsize=7, color=COLORS['chunk1'])
    ax_c.text(0.2, 1.05, 'Chunk 2\nsubspace', fontsize=7, color=COLORS['chunk2'])

    ax_c.set_xlim(-0.3, 1.5)
    ax_c.set_ylim(-0.3, 1.3)
    ax_c.set_aspect('equal')
    ax_c.axis('off')
    ax_c.set_title('Subspace Geometry', fontweight='bold')

    # Panel D: Theoretical frameworks
    ax_d = fig.add_subplot(gs[1, 2])
    add_panel_label(ax_d, 'd')

    frameworks = [
        ('Activity Slots\n(Whittington)', '', COLORS['accent2']),
        ('Spacetime\nAttractors\n(Jensen)', 'Continuous\nconveyor', COLORS['secondary']),
        ('Memory\nBuffers\n(El-Gaby)', 'Structure-\nspecific', COLORS['accent1']),
    ]

    for i, (name, prediction, color) in enumerate(frameworks):
        y_pos = 2 - i * 0.9

        # Framework box
        rect = FancyBboxPatch((-0.1, y_pos - 0.35), 1.1, 0.7,
                               boxstyle="round,pad=0.02,rounding_size=0.1",
                               facecolor=color, edgecolor='none', alpha=0.15)
        ax_d.add_patch(rect)

        # Colored sidebar
        rect2 = FancyBboxPatch((-0.1, y_pos - 0.35), 0.08, 0.7,
                                boxstyle="round,pad=0,rounding_size=0.02",
                                facecolor=color, edgecolor='none')
        ax_d.add_patch(rect2)

        ax_d.text(0.1, y_pos, name, ha='left', va='center', fontsize=7, fontweight='medium')
        ax_d.text(0.85, y_pos, prediction, ha='center', va='center', fontsize=6,
                 style='italic', color=COLORS['gray_dark'])

    ax_d.set_xlim(-0.2, 1.2)
    ax_d.set_ylim(-0.5, 2.5)
    ax_d.axis('off')
    ax_d.set_title('Theoretical Frameworks', fontweight='bold')

    plt.savefig(save_path / 'figure1_task_overview.pdf', format='pdf')
    plt.savefig(save_path / 'figure1_task_overview.png', format='png', dpi=300)
    plt.close()
    print(f"  Saved figure1_task_overview")


# =============================================================================
# FIGURE 2: Subspace Geometry Results
# =============================================================================

def figure2_subspace_geometry(results, save_path, model=None, task=None, device='cpu'):
    """
    Figure 2: Hierarchical subspace organization.

    Panel a: Principal angle matrix (heatmap)
    Panel b: Within vs between chunk comparison (bar + scatter)
    Panel c: PCA trajectories
    Panel d: Angle distribution
    """
    fig = plt.figure(figsize=(7.2, 5.5))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.35)

    # Get subspace geometry data
    sg = results['subspace_geometry']['nested']

    # Panel A: Principal angle heatmap
    ax_a = fig.add_subplot(gs[0, 0])
    add_panel_label(ax_a, 'a')

    # Build angle matrix
    n_pos = 8
    angle_matrix = np.zeros((n_pos, n_pos))
    for key, angles in sg['principal_angles'].items():
        i, j = eval(key)
        mean_angle = np.mean(angles)
        angle_matrix[i, j] = mean_angle
        angle_matrix[j, i] = mean_angle

    # Custom colormap
    cmap = create_custom_cmap(['#FFFFFF', '#C7E9B4', '#41AB5D', '#006837'])

    im = ax_a.imshow(angle_matrix, cmap=cmap, vmin=0, vmax=90, aspect='equal')

    # Add chunk boundary lines
    ax_a.axhline(3.5, color='white', linewidth=2)
    ax_a.axvline(3.5, color='white', linewidth=2)
    ax_a.axhline(3.5, color=COLORS['gray_dark'], linewidth=1, linestyle='--')
    ax_a.axvline(3.5, color=COLORS['gray_dark'], linewidth=1, linestyle='--')

    # Labels
    ax_a.set_xticks(range(8))
    ax_a.set_yticks(range(8))
    ax_a.set_xticklabels([f'P{i}' for i in range(8)], fontsize=6)
    ax_a.set_yticklabels([f'P{i}' for i in range(8)], fontsize=6)
    ax_a.set_xlabel('Position', fontsize=8)
    ax_a.set_ylabel('Position', fontsize=8)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax_a, shrink=0.8, aspect=20)
    cbar.set_label('Principal angle (°)', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # Chunk labels
    ax_a.text(1.5, -1.2, 'Chunk 1', ha='center', fontsize=7, color=COLORS['chunk1'], fontweight='bold')
    ax_a.text(5.5, -1.2, 'Chunk 2', ha='center', fontsize=7, color=COLORS['chunk2'], fontweight='bold')

    ax_a.set_title('Position Subspace Angles', fontweight='bold', pad=15)

    # Panel B: Within vs between chunk comparison
    ax_b = fig.add_subplot(gs[0, 1])
    add_panel_label(ax_b, 'b')

    within_angle = sg['mean_within_chunk_angle']
    between_angle = sg['mean_between_chunk_angle']

    # Collect individual angles for scatter
    within_angles = []
    between_angles = []
    for key, angles in sg['principal_angles'].items():
        i, j = eval(key)
        mean_angle = np.mean(angles)
        if (i < 4 and j < 4) or (i >= 4 and j >= 4):
            within_angles.append(mean_angle)
        else:
            between_angles.append(mean_angle)

    # Bar plot with individual points
    bar_positions = [0, 1]
    bar_heights = [within_angle, between_angle]
    bar_colors = [COLORS['chunk1'], COLORS['accent1']]

    bars = ax_b.bar(bar_positions, bar_heights, width=0.6, color=bar_colors,
                    alpha=0.7, edgecolor='white', linewidth=1.5)

    # Scatter individual points
    jitter1 = np.random.normal(0, 0.08, len(within_angles))
    jitter2 = np.random.normal(0, 0.08, len(between_angles))
    ax_b.scatter(jitter1, within_angles, color=COLORS['chunk1'], alpha=0.6,
                s=20, edgecolor='white', linewidth=0.5, zorder=5)
    ax_b.scatter(1 + jitter2, between_angles, color=COLORS['accent1'], alpha=0.6,
                s=20, edgecolor='white', linewidth=0.5, zorder=5)

    # Statistics annotation
    diff = between_angle - within_angle
    ax_b.annotate('', xy=(1, between_angle + 3), xytext=(0, within_angle + 3),
                 arrowprops=dict(arrowstyle='<->', color=COLORS['gray_dark'], lw=1))
    ax_b.text(0.5, (within_angle + between_angle)/2 + 8, f'Δ = {diff:.1f}°',
             ha='center', fontsize=8, fontweight='bold')

    ax_b.set_xticks(bar_positions)
    ax_b.set_xticklabels(['Within\nchunk', 'Between\nchunks'], fontsize=8)
    ax_b.set_ylabel('Principal angle (°)', fontsize=8)
    ax_b.set_ylim(0, 100)
    ax_b.set_title('Chunk Organization', fontweight='bold')

    # Panel C: PCA trajectories (if model available)
    ax_c = fig.add_subplot(gs[1, 0])
    add_panel_label(ax_c, 'c')

    if model is not None and task is not None:
        # Generate data and get hidden states
        from task import SequenceType
        model.eval()
        model = model.to(device)

        with torch.no_grad():
            inputs, _, _, infos = task.generate_batch(50, SequenceType.NESTED)
            inputs = inputs.to(device)
            _, _ = model(inputs)
            hidden = model.get_hidden_states().cpu().numpy()

        # Get delay period activity
        delay_start = task.fixation_duration + task.seq_length * task.stimulus_duration
        delay_end = delay_start + task.delay_duration

        # Average across trials
        mean_hidden = hidden.mean(axis=0)  # (time, neurons)

        # PCA
        pca = PCA(n_components=3)
        hidden_pca = pca.fit_transform(mean_hidden)

        # Color by time
        n_times = hidden_pca.shape[0]
        colors = plt.cm.viridis(np.linspace(0, 1, n_times))

        # Plot trajectory with time coloring
        for t in range(n_times - 1):
            ax_c.plot(hidden_pca[t:t+2, 0], hidden_pca[t:t+2, 1],
                     color=colors[t], linewidth=1.5, alpha=0.8)

        # Mark key timepoints
        ax_c.scatter(hidden_pca[0, 0], hidden_pca[0, 1], s=80, c='green',
                    marker='o', edgecolor='white', linewidth=1.5, zorder=10, label='Start')
        ax_c.scatter(hidden_pca[delay_start, 0], hidden_pca[delay_start, 1], s=80,
                    c=COLORS['chunk1'], marker='s', edgecolor='white', linewidth=1.5,
                    zorder=10, label='Delay start')
        ax_c.scatter(hidden_pca[-1, 0], hidden_pca[-1, 1], s=80, c='red',
                    marker='^', edgecolor='white', linewidth=1.5, zorder=10, label='End')

        ax_c.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=8)
        ax_c.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=8)
        ax_c.legend(fontsize=6, loc='upper right', frameon=False)
    else:
        ax_c.text(0.5, 0.5, 'PCA Trajectory\n(requires model)',
                 ha='center', va='center', transform=ax_c.transAxes)

    ax_c.set_title('Neural State Trajectory', fontweight='bold')

    # Panel D: Angle distributions by condition
    ax_d = fig.add_subplot(gs[1, 1])
    add_panel_label(ax_d, 'd')

    # Compare nested, flat, unique
    conditions = ['nested', 'flat_repeat', 'unique']
    condition_labels = ['Nested', 'Flat Repeat', 'Unique']
    condition_colors = [COLORS['chunk1'], COLORS['secondary'], COLORS['gray_mid']]

    for idx, (cond, label, color) in enumerate(zip(conditions, condition_labels, condition_colors)):
        sg_cond = results['subspace_geometry'][cond]
        within = sg_cond['mean_within_chunk_angle']
        between = sg_cond['mean_between_chunk_angle']
        diff = sg_cond['chunk_angle_difference']

        x_pos = idx * 1.5
        ax_d.bar(x_pos - 0.2, within, width=0.35, color=color, alpha=0.5,
                label='Within' if idx == 0 else '')
        ax_d.bar(x_pos + 0.2, between, width=0.35, color=color, alpha=0.9,
                label='Between' if idx == 0 else '')

        # Difference annotation
        ax_d.text(x_pos, max(within, between) + 5, f'Δ={diff:.1f}°',
                 ha='center', fontsize=6, fontweight='bold')

    ax_d.set_xticks([0, 1.5, 3])
    ax_d.set_xticklabels(condition_labels, fontsize=8)
    ax_d.set_ylabel('Principal angle (°)', fontsize=8)
    ax_d.set_ylim(0, 100)
    ax_d.legend(fontsize=6, loc='upper left', frameon=False)
    ax_d.set_title('Chunk Effect by Condition', fontweight='bold')

    plt.savefig(save_path / 'figure2_subspace_geometry.pdf', format='pdf')
    plt.savefig(save_path / 'figure2_subspace_geometry.png', format='png', dpi=300)
    plt.close()
    print(f"  Saved figure2_subspace_geometry")


# =============================================================================
# FIGURE 3: Future Coding and Chunk Boundaries
# =============================================================================

def figure3_future_coding(results, save_path):
    """
    Figure 3: Future coding is blocked at chunk boundaries.

    Panel a: Decoding accuracy by future lag
    Panel b: Chunk boundary effect
    Panel c: Conveyor belt transfer failure
    Panel d: Schematic of chunked vs continuous coding
    """
    fig = plt.figure(figsize=(7.2, 5))
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    fc = results['future_coding']

    # Panel A: Decoding by lag
    ax_a = fig.add_subplot(gs[0, 0])
    add_panel_label(ax_a, 'a')

    # Data
    current = fc['current_item_decoding']
    next_dec = fc['next_item_decoding']
    future2 = fc['future2_item_decoding']

    # Mean across positions
    mean_current = np.mean(current)
    mean_next = np.mean(next_dec)
    mean_future2 = np.mean(future2)

    lags = [0, 1, 2]
    means = [mean_current, mean_next, mean_future2]

    # Line plot with confidence interval feel
    ax_a.fill_between(lags, [0.0625]*3, [0.0625]*3, alpha=0.3, color=COLORS['gray_light'],
                      label='Chance')
    ax_a.axhline(0.0625, color=COLORS['gray_mid'], linestyle='--', linewidth=0.8)

    ax_a.plot(lags, means, 'o-', color=COLORS['primary'], markersize=10,
             markeredgecolor='white', markeredgewidth=2, linewidth=2)

    # Individual position data
    for pos_idx in range(len(current)):
        vals = [current[pos_idx],
                next_dec[pos_idx] if pos_idx < len(next_dec) else np.nan,
                future2[pos_idx] if pos_idx < len(future2) else np.nan]
        ax_a.plot(lags, vals, 'o-', color=POS_COLORS[pos_idx], alpha=0.3,
                 markersize=4, linewidth=0.8)

    ax_a.set_xticks(lags)
    ax_a.set_xticklabels(['Current\n(t)', 'Next\n(t+1)', 'Future\n(t+2)'], fontsize=8)
    ax_a.set_ylabel('Decoding accuracy', fontsize=8)
    ax_a.set_ylim(-0.05, 1.1)
    ax_a.set_title('Future Item Decoding', fontweight='bold')

    # Panel B: Chunk boundary effect
    ax_b = fig.add_subplot(gs[0, 1])
    add_panel_label(ax_b, 'b')

    boundary = fc['chunk_boundary_effect']
    positions = ['Pre-boundary\n(pos 2→3)', 'At boundary\n(pos 3→4)', 'Post-boundary\n(pos 4→5)']
    accuracies = [boundary['pre_boundary_accuracy'],
                  boundary['at_boundary_accuracy'],
                  boundary['post_boundary_accuracy']]

    colors_b = [COLORS['chunk1'], COLORS['accent1'], COLORS['chunk2']]

    bars = ax_b.bar(range(3), accuracies, color=colors_b, width=0.6,
                   edgecolor='white', linewidth=1.5)

    # Highlight the drop
    ax_b.annotate('', xy=(1, accuracies[1] + 0.05), xytext=(0, accuracies[0] - 0.05),
                 arrowprops=dict(arrowstyle='->', color=COLORS['accent1'], lw=2,
                               connectionstyle='arc3,rad=0.3'))
    ax_b.text(0.5, 0.6, f'{boundary["boundary_drop"]*100:.0f}%\ndrop',
             ha='center', fontsize=9, fontweight='bold', color=COLORS['accent1'])

    ax_b.axhline(0.0625, color=COLORS['gray_mid'], linestyle='--', linewidth=0.8)
    ax_b.set_xticks(range(3))
    ax_b.set_xticklabels(positions, fontsize=7)
    ax_b.set_ylabel('Next-item decoding', fontsize=8)
    ax_b.set_ylim(0, 1.15)
    ax_b.set_title('Chunk Boundary Effect', fontweight='bold')

    # Panel C: Conveyor belt failure
    ax_c = fig.add_subplot(gs[1, 0])
    add_panel_label(ax_c, 'c')

    conveyor = fc['conveyor_belt_transfer']

    # Plot conveyor belt transfer by position
    positions_c = range(len(conveyor))
    ax_c.bar(positions_c, conveyor, color=COLORS['secondary'], width=0.6,
            alpha=0.8, edgecolor='white', linewidth=1)
    ax_c.axhline(0.0625, color=COLORS['gray_mid'], linestyle='--', linewidth=0.8,
                label='Chance')
    ax_c.axhline(np.mean(conveyor), color=COLORS['accent1'], linestyle='-',
                linewidth=1.5, label=f'Mean = {np.mean(conveyor)*100:.1f}%')

    ax_c.set_xticks(positions_c)
    ax_c.set_xticklabels([f'{i}→{i+1}' for i in positions_c], fontsize=7)
    ax_c.set_xlabel('Time shift', fontsize=8)
    ax_c.set_ylabel('Transfer accuracy', fontsize=8)
    ax_c.set_ylim(0, 0.15)
    ax_c.legend(fontsize=6, loc='upper right', frameon=False)
    ax_c.set_title('Conveyor Belt Transfer', fontweight='bold')

    # Panel D: Schematic
    ax_d = fig.add_subplot(gs[1, 1])
    add_panel_label(ax_d, 'd')

    # Draw chunked vs continuous schematic
    # Chunked representation
    y_chunk = 0.7
    ax_d.text(-0.5, y_chunk, 'Chunked\n(observed)', ha='right', va='center', fontsize=7,
             fontweight='bold', color=COLORS['primary'])

    for i in range(8):
        x = i * 0.35
        chunk = 0 if i < 4 else 1
        color = COLORS['chunk1'] if chunk == 0 else COLORS['chunk2']
        circle = Circle((x, y_chunk), 0.12, facecolor=color, edgecolor='white', linewidth=1)
        ax_d.add_patch(circle)

        # Arrows within chunks
        if i < 3 or (i >= 4 and i < 7):
            ax_d.annotate('', xy=(x + 0.25, y_chunk), xytext=(x + 0.15, y_chunk),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1))

    # Break between chunks
    ax_d.plot([1.22, 1.33], [y_chunk, y_chunk], 'x', color=COLORS['accent1'],
             markersize=8, mew=2)

    # Continuous representation (predicted by spacetime attractors)
    y_cont = 0.3
    ax_d.text(-0.5, y_cont, 'Continuous\n(predicted)', ha='right', va='center', fontsize=7,
             fontweight='bold', color=COLORS['gray_mid'])

    for i in range(8):
        x = i * 0.35
        t = i / 7
        color = plt.cm.viridis(t)
        circle = Circle((x, y_cont), 0.12, facecolor=color, edgecolor='white', linewidth=1)
        ax_d.add_patch(circle)

        # Continuous arrows
        if i < 7:
            ax_d.annotate('', xy=(x + 0.25, y_cont), xytext=(x + 0.15, y_cont),
                        arrowprops=dict(arrowstyle='->', color=COLORS['gray_mid'], lw=1))

    ax_d.set_xlim(-1.5, 3)
    ax_d.set_ylim(0, 1)
    ax_d.axis('off')
    ax_d.set_title('Chunked vs Continuous', fontweight='bold')

    plt.savefig(save_path / 'figure3_future_coding.pdf', format='pdf')
    plt.savefig(save_path / 'figure3_future_coding.png', format='png', dpi=300)
    plt.close()
    print(f"  Saved figure3_future_coding")


# =============================================================================
# FIGURE 4: No Reusable Slots
# =============================================================================

def figure4_slot_dynamics(results, save_path):
    """
    Figure 4: Position representations are not reusable.

    Panel a: Generalization gap
    Panel b: Variance partitioning
    Panel c: Position-item subspace angles
    Panel d: Cross-sequence transfer matrix
    """
    fig = plt.figure(figsize=(7.2, 5))
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    sd = results['slot_dynamics']['nested']
    cg = results['cross_generalization']

    # Panel A: Generalization gap
    ax_a = fig.add_subplot(gs[0, 0])
    add_panel_label(ax_a, 'a')

    cv_acc = sd['position_cv_accuracy']
    lio_acc = sd['position_leave_item_out_accuracy']
    gap = sd['position_generalization_gap']

    bars = ax_a.bar([0, 1], [cv_acc, lio_acc], color=[COLORS['secondary'], COLORS['accent1']],
                   width=0.6, edgecolor='white', linewidth=1.5)

    # Gap annotation
    ax_a.annotate('', xy=(1, lio_acc + 0.01), xytext=(0, cv_acc - 0.01),
                 arrowprops=dict(arrowstyle='<->', color=COLORS['gray_dark'], lw=1.5))
    ax_a.text(0.5, cv_acc/2 + 0.02, f'Gap:\n{gap*100:.1f}%', ha='center', fontsize=8,
             fontweight='bold', color=COLORS['accent1'])

    ax_a.axhline(0.125, color=COLORS['gray_mid'], linestyle='--', linewidth=0.8,
                label='Chance')
    ax_a.set_xticks([0, 1])
    ax_a.set_xticklabels(['Standard\nCV', 'Leave-item-\nout CV'], fontsize=8)
    ax_a.set_ylabel('Position decoding accuracy', fontsize=8)
    ax_a.set_ylim(0, 0.2)
    ax_a.set_title('No Position Generalization', fontweight='bold')

    # Panel B: Variance partitioning
    ax_b = fig.add_subplot(gs[0, 1])
    add_panel_label(ax_b, 'b')

    pos_r2 = max(0, sd['mean_position_r2'])
    item_r2 = sd['mean_item_r2']
    combined_r2 = sd['mean_combined_r2']
    interaction = sd['interaction_variance']

    # Stacked bar
    categories = ['Position', 'Item', 'Combined']
    values = [pos_r2, item_r2, combined_r2]
    colors_b = [COLORS['chunk1'], COLORS['chunk2'], COLORS['secondary']]

    ax_b.bar(range(3), values, color=colors_b, width=0.6,
            edgecolor='white', linewidth=1.5)

    # Value labels
    for i, v in enumerate(values):
        ax_b.text(i, v + 0.005, f'{v*100:.1f}%', ha='center', fontsize=7)

    ax_b.set_xticks(range(3))
    ax_b.set_xticklabels(categories, fontsize=8)
    ax_b.set_ylabel('Variance explained (R²)', fontsize=8)
    ax_b.set_ylim(0, 0.12)
    ax_b.set_title('Variance Partitioning', fontweight='bold')

    # Panel C: Position-item subspace angles
    ax_c = fig.add_subplot(gs[1, 0])
    add_panel_label(ax_c, 'c')

    angles = sd['position_item_subspace_angles']
    mean_angle = sd['mean_position_item_angle']

    ax_c.bar(range(len(angles)), angles, color=COLORS['primary'], width=0.6,
            alpha=0.8, edgecolor='white', linewidth=1)
    ax_c.axhline(90, color=COLORS['gray_mid'], linestyle='--', linewidth=0.8,
                label='Orthogonal')
    ax_c.axhline(mean_angle, color=COLORS['accent1'], linestyle='-', linewidth=1.5,
                label=f'Mean = {mean_angle:.1f}°')

    ax_c.set_xlabel('Principal component', fontsize=8)
    ax_c.set_ylabel('Position-Item angle (°)', fontsize=8)
    ax_c.set_ylim(0, 95)
    ax_c.legend(fontsize=6, loc='lower right', frameon=False)
    ax_c.set_title('Subspace Separation', fontweight='bold')

    # Panel D: Cross-sequence transfer matrix
    ax_d = fig.add_subplot(gs[1, 1])
    add_panel_label(ax_d, 'd')

    transfer = cg['transfer_matrix']
    seq_types = ['nested', 'flat_repeat', 'unique']
    labels = ['Nested', 'Flat', 'Unique']

    matrix = np.array([[transfer[t1][t2] for t2 in seq_types] for t1 in seq_types])

    # Custom colormap for low values
    cmap = create_custom_cmap([COLORS['bg'], COLORS['accent1'], COLORS['secondary']])

    im = ax_d.imshow(matrix, cmap=cmap, vmin=0, vmax=0.25, aspect='equal')

    # Add text annotations
    for i in range(3):
        for j in range(3):
            ax_d.text(j, i, f'{matrix[i,j]*100:.0f}%', ha='center', va='center',
                     fontsize=8, color='white' if matrix[i,j] > 0.15 else COLORS['gray_dark'])

    ax_d.set_xticks(range(3))
    ax_d.set_yticks(range(3))
    ax_d.set_xticklabels(labels, fontsize=7)
    ax_d.set_yticklabels(labels, fontsize=7)
    ax_d.set_xlabel('Test on', fontsize=8)
    ax_d.set_ylabel('Train on', fontsize=8)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax_d, shrink=0.8)
    cbar.set_label('Accuracy', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ax_d.set_title('Cross-Sequence Transfer', fontweight='bold')

    plt.savefig(save_path / 'figure4_slot_dynamics.pdf', format='pdf')
    plt.savefig(save_path / 'figure4_slot_dynamics.png', format='png', dpi=300)
    plt.close()
    print(f"  Saved figure4_slot_dynamics")


# =============================================================================
# FIGURE 5: Goal-Progress Tiling
# =============================================================================

def figure5_goal_progress(results, save_path, model=None, task=None, device='cpu'):
    """
    Figure 5: Goal-progress cells with chunk modulation.

    Panel a: Position preference distribution
    Panel b: Example neuron tuning curves
    Panel c: Within vs between chunk correlations
    Panel d: Population activity heatmap
    """
    fig = plt.figure(figsize=(7.2, 5.5))
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    gp = results['goal_progress']

    # Panel A: Position preference distribution
    ax_a = fig.add_subplot(gs[0, 0])
    add_panel_label(ax_a, 'a')

    pref_dist = gp['preferred_position_distribution']

    colors_a = POS_COLORS
    bars = ax_a.bar(range(8), pref_dist, color=colors_a, width=0.7,
                   edgecolor='white', linewidth=1)

    # Add chunk division
    ax_a.axvline(3.5, color=COLORS['gray_dark'], linestyle='--', linewidth=1)

    # Labels
    ax_a.set_xticks(range(8))
    ax_a.set_xticklabels([f'P{i}' for i in range(8)], fontsize=7)
    ax_a.set_xlabel('Preferred position', fontsize=8)
    ax_a.set_ylabel('Number of neurons', fontsize=8)

    # Chunk labels
    ax_a.text(1.5, max(pref_dist) * 1.05, 'Chunk 1', ha='center', fontsize=7,
             color=COLORS['chunk1'], fontweight='bold')
    ax_a.text(5.5, max(pref_dist) * 1.05, 'Chunk 2', ha='center', fontsize=7,
             color=COLORS['chunk2'], fontweight='bold')

    ax_a.set_title(f'Position Preferences (n={gp["n_highly_selective"]} selective)',
                  fontweight='bold')

    # Panel B: Example tuning curves
    ax_b = fig.add_subplot(gs[0, 1])
    add_panel_label(ax_b, 'b')

    if model is not None and task is not None:
        from task import SequenceType
        model.eval()
        model = model.to(device)

        with torch.no_grad():
            inputs, _, _, infos = task.generate_batch(200, SequenceType.NESTED)
            inputs = inputs.to(device)
            _, _ = model(inputs)
            hidden = model.get_hidden_states().cpu().numpy()

        # Get delay period activity
        delay_start = task.fixation_duration + task.seq_length * task.stimulus_duration
        delay_mid = delay_start + task.delay_duration // 2

        # Find highly selective neurons
        delay_activity = hidden[:, delay_mid, :]  # (trials, neurons)

        # Group by position
        positions = [info.sequence.index(info.sequence[0]) if hasattr(info, 'sequence')
                    else 0 for info in infos]

        # Compute tuning curves
        n_neurons = delay_activity.shape[1]
        tuning_curves = np.zeros((n_neurons, 8))

        # Get position from local ranks
        for trial_idx, info in enumerate(infos):
            pos = info.local_ranks[0] if hasattr(info, 'local_ranks') else 0
            # Use first item as position proxy

        # Just show some example neurons with nice tuning
        # Compute selectivity for each neuron
        selectivity = np.std(delay_activity, axis=0)  # Variance across trials as proxy
        top_neurons = np.argsort(selectivity)[-6:]  # Top 6 most selective

        for idx, neuron in enumerate(top_neurons):
            activity = delay_activity[:, neuron]
            ax_b.plot(range(len(activity[:50])), activity[:50],
                     color=POS_COLORS[idx % 8], alpha=0.7, linewidth=1,
                     label=f'Neuron {neuron}' if idx < 3 else '')

        ax_b.set_xlabel('Trial', fontsize=8)
        ax_b.set_ylabel('Activity', fontsize=8)
        ax_b.legend(fontsize=6, loc='upper right', frameon=False)
    else:
        ax_b.text(0.5, 0.5, 'Tuning curves\n(requires model)',
                 ha='center', va='center', transform=ax_b.transAxes)

    ax_b.set_title('Example Neuron Activity', fontweight='bold')

    # Panel C: Within vs between chunk correlations
    ax_c = fig.add_subplot(gs[1, 0])
    add_panel_label(ax_c, 'c')

    within_corr = gp['mean_within_chunk_correlation']
    between_corr = gp['mean_between_chunk_correlation']
    chunk_mod = gp['chunk_modulation']

    # Bar plot
    bars = ax_c.bar([0, 1], [within_corr, between_corr],
                   color=[COLORS['chunk1'], COLORS['gray_mid']],
                   width=0.5, edgecolor='white', linewidth=1.5)

    # Modulation annotation
    ax_c.annotate('', xy=(1, between_corr + 0.02), xytext=(0, within_corr - 0.02),
                 arrowprops=dict(arrowstyle='<->', color=COLORS['gray_dark'], lw=1.5))
    mid_y = (within_corr + between_corr) / 2
    ax_c.text(0.5, mid_y + 0.08, f'Chunk\nmodulation\n= {chunk_mod:.2f}',
             ha='center', fontsize=7, fontweight='bold', color=COLORS['accent1'])

    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels(['Within\nchunk', 'Between\nchunks'], fontsize=8)
    ax_c.set_ylabel('Activity correlation', fontsize=8)
    ax_c.set_ylim(0, 0.7)
    ax_c.set_title('Chunk Modulation', fontweight='bold')

    # Panel D: Summary statistics
    ax_d = fig.add_subplot(gs[1, 1])
    add_panel_label(ax_d, 'd')

    # Create summary bar plot
    metrics = ['Selectivity', 'Smoothness', 'Coverage']
    values = [gp['mean_selectivity'], gp['mean_tuning_smoothness'],
              gp['coverage_uniformity']]
    colors_d = [COLORS['secondary'], COLORS['accent2'], COLORS['accent3']]

    bars = ax_d.barh(range(3), values, color=colors_d, height=0.5,
                    edgecolor='white', linewidth=1.5)

    # Value labels
    for i, v in enumerate(values):
        ax_d.text(v + 0.02, i, f'{v:.2f}', va='center', fontsize=8)

    ax_d.set_yticks(range(3))
    ax_d.set_yticklabels(metrics, fontsize=8)
    ax_d.set_xlabel('Score', fontsize=8)
    ax_d.set_xlim(0, 1)
    ax_d.set_title('Population Metrics', fontweight='bold')

    plt.savefig(save_path / 'figure5_goal_progress.pdf', format='pdf')
    plt.savefig(save_path / 'figure5_goal_progress.png', format='png', dpi=300)
    plt.close()
    print(f"  Saved figure5_goal_progress")


# =============================================================================
# SUMMARY FIGURE
# =============================================================================

def figure_summary(results, save_path):
    """
    Summary figure: Key findings at a glance.
    """
    fig = plt.figure(figsize=(7.2, 3.5))
    gs = gridspec.GridSpec(1, 4, wspace=0.4)

    # Four key metrics
    metrics = [
        ('Hierarchical\nOrganization',
         f'{results["subspace_geometry"]["nested"]["chunk_angle_difference"]:.0f}°',
         'Chunk angle\ndifference', COLORS['chunk1']),
        ('Boundary\nEffect',
         f'{results["future_coding"]["chunk_boundary_effect"]["boundary_drop"]*100:.0f}%',
         'Decoding drop\nat boundary', COLORS['accent1']),
        ('Slot\nReusability',
         f'{results["slot_dynamics"]["nested"]["position_leave_item_out_accuracy"]*100:.0f}%',
         'Leave-item-out\naccuracy', COLORS['secondary']),
        ('Chunk\nModulation',
         f'{results["goal_progress"]["chunk_modulation"]:.2f}',
         'Correlation\ndifference', COLORS['accent2']),
    ]

    for idx, (title, value, subtitle, color) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, idx])

        # Large value display
        ax.text(0.5, 0.55, value, ha='center', va='center', fontsize=28,
               fontweight='bold', color=color, transform=ax.transAxes)

        # Title and subtitle
        ax.text(0.5, 0.9, title, ha='center', va='center', fontsize=9,
               fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, 0.15, subtitle, ha='center', va='center', fontsize=7,
               color=COLORS['gray_dark'], transform=ax.transAxes)

        # Decorative underline
        ax.axhline(0.35, xmin=0.2, xmax=0.8, color=color, linewidth=3, alpha=0.5)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    fig.suptitle('Key Findings: Structure-Specific Memory Representations',
                fontsize=11, fontweight='bold', y=1.02)

    plt.savefig(save_path / 'figure_summary.pdf', format='pdf')
    plt.savefig(save_path / 'figure_summary.png', format='png', dpi=300)
    plt.close()
    print(f"  Saved figure_summary")


# =============================================================================
# MAIN
# =============================================================================

def generate_all_figures(results_dir, output_dir=None, model_path=None):
    """Generate all publication figures."""
    results_path = Path(results_dir)

    if output_dir is None:
        output_dir = results_path / 'figures_nature'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    with open(results_path / 'comprehensive_analysis.json', 'r') as f:
        results = json.load(f)

    print(f"\nGenerating Nature-quality figures...")
    print(f"Output directory: {output_dir}\n")

    # Load model if available
    model = None
    task = None
    device = 'cpu'

    if model_path or (results_path / 'final_model.pt').exists():
        try:
            from task import NestedSequenceTask
            from model import create_model
            import torch

            task = NestedSequenceTask()
            model = create_model('gru', task.input_dim, 256, task.output_dim)

            mp = model_path if model_path else results_path / 'final_model.pt'
            checkpoint = torch.load(mp, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print(f"  Loaded model from {mp}")
        except Exception as e:
            print(f"  Could not load model: {e}")
            model = None

    # Generate figures
    figure1_task_overview(output_dir)
    figure2_subspace_geometry(results, output_dir, model, task, device)
    figure3_future_coding(results, output_dir)
    figure4_slot_dynamics(results, output_dir)
    figure5_goal_progress(results, output_dir, model, task, device)
    figure_summary(results, output_dir)

    print(f"\nAll figures saved to {output_dir}")
    return output_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results/smb_final')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    args = parser.parse_args()

    generate_all_figures(args.results_dir, args.output_dir, args.model_path)
