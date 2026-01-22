#!/usr/bin/env python
"""
Graphical Abstract for Nature-style presentation.

A single, visually striking figure that captures the essence of the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Wedge, Arc
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pathlib import Path
import json

# Publication defaults
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.linewidth': 0,
    'figure.dpi': 150,
    'savefig.dpi': 400,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2,
})

# Sophisticated color palette
COLORS = {
    'bg': '#0D1117',           # Dark background
    'bg_light': '#161B22',
    'chunk1': '#58A6FF',        # Blue
    'chunk2': '#F778BA',        # Pink
    'accent': '#7EE787',        # Green
    'warning': '#FFA657',       # Orange
    'text': '#E6EDF3',
    'text_dim': '#8B949E',
    'white': '#FFFFFF',
    'gradient_start': '#238636',
    'gradient_end': '#1F6FEB',
}


def create_graphical_abstract(results_path, output_path):
    """Create a visually striking graphical abstract."""

    # Load results
    with open(results_path / 'comprehensive_analysis.json', 'r') as f:
        results = json.load(f)

    # Create figure with dark background
    fig = plt.figure(figsize=(10, 8), facecolor=COLORS['bg'])

    # Main title
    fig.text(0.5, 0.95, 'Structured Memory Without Reusable Slots',
             ha='center', va='top', fontsize=18, fontweight='bold',
             color=COLORS['white'])
    fig.text(0.5, 0.91, 'How RNNs Represent Hierarchical Sequences',
             ha='center', va='top', fontsize=12, color=COLORS['text_dim'])

    # Create custom grid
    gs = gridspec.GridSpec(3, 3, figure=fig,
                           left=0.08, right=0.92, top=0.85, bottom=0.08,
                           hspace=0.35, wspace=0.3)

    # =========================================================================
    # TOP ROW: Task and Key Question
    # =========================================================================

    # Panel: Sequence structure
    ax_seq = fig.add_subplot(gs[0, 0], facecolor=COLORS['bg'])
    draw_sequence_structure(ax_seq)

    # Panel: The question
    ax_q = fig.add_subplot(gs[0, 1], facecolor=COLORS['bg'])
    draw_question(ax_q)

    # Panel: Three theories
    ax_theories = fig.add_subplot(gs[0, 2], facecolor=COLORS['bg'])
    draw_theories(ax_theories)

    # =========================================================================
    # MIDDLE ROW: Key Findings
    # =========================================================================

    # Panel: Hierarchical geometry
    ax_geom = fig.add_subplot(gs[1, 0], facecolor=COLORS['bg'])
    draw_geometry_result(ax_geom, results)

    # Panel: Chunk boundary
    ax_boundary = fig.add_subplot(gs[1, 1], facecolor=COLORS['bg'])
    draw_boundary_result(ax_boundary, results)

    # Panel: No reusable slots
    ax_slots = fig.add_subplot(gs[1, 2], facecolor=COLORS['bg'])
    draw_slots_result(ax_slots, results)

    # =========================================================================
    # BOTTOM ROW: Conclusion
    # =========================================================================

    ax_conclusion = fig.add_subplot(gs[2, :], facecolor=COLORS['bg'])
    draw_conclusion(ax_conclusion, results)

    # Save
    plt.savefig(output_path / 'graphical_abstract.pdf', format='pdf',
                facecolor=COLORS['bg'], edgecolor='none')
    plt.savefig(output_path / 'graphical_abstract.png', format='png',
                facecolor=COLORS['bg'], edgecolor='none', dpi=400)
    plt.close()
    print(f"Saved graphical abstract to {output_path}")


def draw_sequence_structure(ax):
    """Draw the nested sequence structure."""
    ax.set_title('Nested Sequences', color=COLORS['white'], fontsize=11,
                fontweight='bold', pad=10)

    # Draw ABABCDCD pattern
    items = ['A', 'B', 'A', 'B', 'C', 'D', 'C', 'D']
    colors = [COLORS['chunk1']] * 4 + [COLORS['chunk2']] * 4

    for i, (item, color) in enumerate(zip(items, colors)):
        x = i * 0.12 + 0.05
        # Glow effect
        circle_glow = Circle((x, 0.5), 0.055, facecolor=color, alpha=0.3)
        ax.add_patch(circle_glow)
        # Main circle
        circle = Circle((x, 0.5), 0.04, facecolor=color, edgecolor=COLORS['white'],
                        linewidth=0.5)
        ax.add_patch(circle)
        ax.text(x, 0.5, item, ha='center', va='center', fontsize=8,
               color=COLORS['white'], fontweight='bold')

    # Chunk brackets
    ax.annotate('', xy=(0.45, 0.7), xytext=(0.05, 0.7),
               arrowprops=dict(arrowstyle='-', color=COLORS['chunk1'], lw=2))
    ax.annotate('', xy=(0.95, 0.7), xytext=(0.55, 0.7),
               arrowprops=dict(arrowstyle='-', color=COLORS['chunk2'], lw=2))
    ax.text(0.25, 0.78, 'Chunk 1', ha='center', fontsize=7, color=COLORS['chunk1'])
    ax.text(0.75, 0.78, 'Chunk 2', ha='center', fontsize=7, color=COLORS['chunk2'])

    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 0.9)
    ax.axis('off')


def draw_question(ax):
    """Draw the central question."""
    ax.set_title('The Question', color=COLORS['white'], fontsize=11,
                fontweight='bold', pad=10)

    # Question text
    ax.text(0.5, 0.65, 'How do neural networks\nrepresent hierarchical\nsequence structure?',
           ha='center', va='center', fontsize=10, color=COLORS['white'],
           linespacing=1.3)

    # Decorative element
    angles = np.linspace(0, 2*np.pi, 100)
    r = 0.35
    ax.plot(0.5 + r * np.cos(angles), 0.55 + r * 0.5 * np.sin(angles),
           color=COLORS['accent'], alpha=0.3, linewidth=2)

    ax.text(0.5, 0.25, '?', ha='center', va='center', fontsize=36,
           color=COLORS['accent'], fontweight='bold', alpha=0.5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def draw_theories(ax):
    """Draw the three competing theories."""
    ax.set_title('Competing Theories', color=COLORS['white'], fontsize=11,
                fontweight='bold', pad=10)

    theories = [
        ('Slots', 'Reusable scaffolds', COLORS['warning'], 0.75),
        ('Attractors', 'Continuous conveyor', COLORS['chunk1'], 0.5),
        ('Buffers', 'Structure-specific', COLORS['accent'], 0.25),
    ]

    for name, desc, color, y in theories:
        # Colored indicator
        rect = FancyBboxPatch((0.05, y - 0.08), 0.03, 0.16,
                               boxstyle="round,pad=0.01",
                               facecolor=color, edgecolor='none')
        ax.add_patch(rect)

        ax.text(0.12, y, name, ha='left', va='center', fontsize=9,
               color=COLORS['white'], fontweight='bold')
        ax.text(0.12, y - 0.08, desc, ha='left', va='top', fontsize=7,
               color=COLORS['text_dim'])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


def draw_geometry_result(ax, results):
    """Draw the hierarchical geometry finding."""
    ax.set_title('Finding 1: Hierarchical Geometry', color=COLORS['white'],
                fontsize=10, fontweight='bold', pad=8)

    sg = results['subspace_geometry']['nested']
    within = sg['mean_within_chunk_angle']
    between = sg['mean_between_chunk_angle']
    diff = sg['chunk_angle_difference']

    # Draw two subspaces as planes
    # Chunk 1 subspace (horizontal-ish)
    theta1 = np.radians(20)
    x1 = [0.2, 0.2 + 0.5 * np.cos(theta1)]
    y1 = [0.35, 0.35 + 0.5 * np.sin(theta1)]
    ax.plot(x1, y1, color=COLORS['chunk1'], linewidth=4, alpha=0.8)
    ax.plot([x1[0], x1[0] + 0.3 * np.cos(theta1 + np.pi/2)],
           [y1[0], y1[0] + 0.3 * np.sin(theta1 + np.pi/2)],
           color=COLORS['chunk1'], linewidth=3, alpha=0.5)

    # Chunk 2 subspace (nearly orthogonal)
    theta2 = np.radians(75)
    x2 = [0.2, 0.2 + 0.5 * np.cos(theta2)]
    y2 = [0.35, 0.35 + 0.5 * np.sin(theta2)]
    ax.plot(x2, y2, color=COLORS['chunk2'], linewidth=4, alpha=0.8)
    ax.plot([x2[0], x2[0] + 0.3 * np.cos(theta2 + np.pi/2)],
           [y2[0], y2[0] + 0.3 * np.sin(theta2 + np.pi/2)],
           color=COLORS['chunk2'], linewidth=3, alpha=0.5)

    # Angle arc
    angles = np.linspace(theta1, theta2, 30)
    arc_r = 0.25
    ax.plot(0.2 + arc_r * np.cos(angles), 0.35 + arc_r * np.sin(angles),
           color=COLORS['white'], linewidth=1.5, alpha=0.7)

    # Large number display
    ax.text(0.75, 0.55, f'{diff:.0f}°', ha='center', va='center', fontsize=24,
           color=COLORS['accent'], fontweight='bold')
    ax.text(0.75, 0.35, 'angle\ndifference', ha='center', va='center', fontsize=8,
           color=COLORS['text_dim'], linespacing=1.2)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.85)
    ax.axis('off')


def draw_boundary_result(ax, results):
    """Draw the chunk boundary effect."""
    ax.set_title('Finding 2: Boundary Block', color=COLORS['white'],
                fontsize=10, fontweight='bold', pad=8)

    fc = results['future_coding']
    drop = fc['chunk_boundary_effect']['boundary_drop']

    # Draw timeline with boundary
    y_line = 0.4
    for i in range(8):
        x = 0.1 + i * 0.1
        color = COLORS['chunk1'] if i < 4 else COLORS['chunk2']

        # Position indicator
        circle = Circle((x, y_line), 0.025, facecolor=color, edgecolor='none')
        ax.add_patch(circle)

        # Arrows between positions (except at boundary)
        if i < 7 and i != 3:
            ax.annotate('', xy=(x + 0.07, y_line), xytext=(x + 0.03, y_line),
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    # X at boundary
    ax.text(0.45, y_line, 'X', ha='center', va='center', fontsize=14,
           color=COLORS['warning'], fontweight='bold')

    # Drop percentage
    ax.text(0.75, 0.65, f'{drop*100:.0f}%', ha='center', va='center', fontsize=24,
           color=COLORS['warning'], fontweight='bold')
    ax.text(0.75, 0.45, 'decoding\ndrop', ha='center', va='center', fontsize=8,
           color=COLORS['text_dim'], linespacing=1.2)

    # Labels
    ax.text(0.3, 0.2, 'Chunk 1', ha='center', fontsize=7, color=COLORS['chunk1'])
    ax.text(0.65, 0.2, 'Chunk 2', ha='center', fontsize=7, color=COLORS['chunk2'])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.85)
    ax.axis('off')


def draw_slots_result(ax, results):
    """Draw the no reusable slots finding."""
    ax.set_title('Finding 3: No Reusable Slots', color=COLORS['white'],
                fontsize=10, fontweight='bold', pad=8)

    sd = results['slot_dynamics']['nested']
    gap = sd['position_generalization_gap']
    lio = sd['position_leave_item_out_accuracy']

    # Draw "slot" metaphor with X through it
    # Empty slot frames
    for i in range(4):
        x = 0.15 + i * 0.18
        rect = FancyBboxPatch((x, 0.35), 0.12, 0.25,
                               boxstyle="round,pad=0.02",
                               facecolor='none', edgecolor=COLORS['text_dim'],
                               linewidth=1, linestyle='--')
        ax.add_patch(rect)
        ax.text(x + 0.06, 0.47, '?', ha='center', va='center',
               fontsize=12, color=COLORS['text_dim'])

    # Big X over slots
    ax.plot([0.1, 0.85], [0.3, 0.65], color=COLORS['warning'], linewidth=3, alpha=0.8)
    ax.plot([0.1, 0.85], [0.65, 0.3], color=COLORS['warning'], linewidth=3, alpha=0.8)

    # Percentage
    ax.text(0.5, 0.15, f'{lio*100:.0f}% generalization', ha='center', va='center',
           fontsize=10, color=COLORS['warning'], fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.85)
    ax.axis('off')


def draw_conclusion(ax, results):
    """Draw the conclusion panel."""

    # Central conclusion
    conclusion_text = (
        'RNNs develop structure-specific, conjunctive representations '
        'rather than factorized slots'
    )

    # Background glow
    rect = FancyBboxPatch((0.1, 0.35), 0.8, 0.4,
                           boxstyle="round,pad=0.03",
                           facecolor=COLORS['bg_light'],
                           edgecolor=COLORS['accent'], linewidth=2, alpha=0.5)
    ax.add_patch(rect)

    ax.text(0.5, 0.55, conclusion_text, ha='center', va='center',
           fontsize=12, color=COLORS['white'], fontweight='bold',
           wrap=True)

    # Winner indication
    ax.text(0.5, 0.85, 'Consistent with Memory Buffers (El-Gaby et al.)',
           ha='center', va='center', fontsize=10, color=COLORS['accent'],
           fontweight='bold')

    # Icons for rejected theories
    ax.text(0.2, 0.15, 'Slots', ha='center', va='center', fontsize=9,
           color=COLORS['warning'], alpha=0.6)
    ax.plot([0.15, 0.25], [0.15, 0.15], color=COLORS['warning'], linewidth=2, alpha=0.6)
    ax.text(0.35, 0.15, 'X', ha='center', va='center', fontsize=10,
           color=COLORS['warning'], fontweight='bold')

    ax.text(0.65, 0.15, 'Attractors', ha='center', va='center', fontsize=9,
           color=COLORS['chunk1'], alpha=0.6)
    ax.plot([0.55, 0.75], [0.15, 0.15], color=COLORS['chunk1'], linewidth=2, alpha=0.6)
    ax.text(0.85, 0.15, 'X', ha='center', va='center', fontsize=10,
           color=COLORS['chunk1'], fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


# Light mode version
def create_graphical_abstract_light(results_path, output_path):
    """Create a light mode version for print."""

    COLORS_LIGHT = {
        'bg': '#FFFFFF',
        'bg_light': '#F6F8FA',
        'chunk1': '#0969DA',
        'chunk2': '#BF3989',
        'accent': '#1A7F37',
        'warning': '#9A6700',
        'text': '#24292F',
        'text_dim': '#57606A',
        'white': '#24292F',
    }

    # Load results
    with open(results_path / 'comprehensive_analysis.json', 'r') as f:
        results = json.load(f)

    fig = plt.figure(figsize=(10, 8), facecolor=COLORS_LIGHT['bg'])

    # Title
    fig.text(0.5, 0.95, 'Structured Memory Without Reusable Slots',
             ha='center', va='top', fontsize=18, fontweight='bold',
             color=COLORS_LIGHT['text'])
    fig.text(0.5, 0.91, 'How RNNs Represent Hierarchical Sequences',
             ha='center', va='top', fontsize=12, color=COLORS_LIGHT['text_dim'])

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           left=0.08, right=0.92, top=0.85, bottom=0.08,
                           hspace=0.35, wspace=0.3)

    # Key metrics panel
    ax_metrics = fig.add_subplot(gs[:, :], facecolor=COLORS_LIGHT['bg'])

    # Create elegant metric cards
    metrics = [
        (0.15, 0.65, '36.8°', 'Hierarchical\nOrganization', COLORS_LIGHT['chunk1'],
         'Chunk angle difference'),
        (0.4, 0.65, '93%', 'Boundary\nEffect', COLORS_LIGHT['warning'],
         'Decoding drop at boundary'),
        (0.65, 0.65, '0%', 'Slot\nReusability', COLORS_LIGHT['accent'],
         'Leave-item-out accuracy'),
        (0.9, 0.65, '0.20', 'Chunk\nModulation', COLORS_LIGHT['chunk2'],
         'Correlation difference'),
    ]

    for x, y, value, title, color, subtitle in metrics:
        # Card background
        rect = FancyBboxPatch((x - 0.1, y - 0.25), 0.2, 0.45,
                               boxstyle="round,pad=0.02",
                               facecolor=COLORS_LIGHT['bg_light'],
                               edgecolor=color, linewidth=2)
        ax_metrics.add_patch(rect)

        # Value
        ax_metrics.text(x, y + 0.05, value, ha='center', va='center',
                       fontsize=28, color=color, fontweight='bold')

        # Title
        ax_metrics.text(x, y - 0.12, title, ha='center', va='center',
                       fontsize=10, color=COLORS_LIGHT['text'],
                       fontweight='bold', linespacing=1.1)

    # Conclusion box
    rect = FancyBboxPatch((0.1, 0.05), 0.8, 0.25,
                           boxstyle="round,pad=0.02",
                           facecolor=COLORS_LIGHT['bg_light'],
                           edgecolor=COLORS_LIGHT['accent'], linewidth=2)
    ax_metrics.add_patch(rect)

    ax_metrics.text(0.5, 0.22, 'Conclusion:', ha='center', va='center',
                   fontsize=11, color=COLORS_LIGHT['accent'], fontweight='bold')
    ax_metrics.text(0.5, 0.12, 'RNNs develop structure-specific representations\n'
                    'consistent with Memory Buffers (El-Gaby et al.)',
                   ha='center', va='center', fontsize=10,
                   color=COLORS_LIGHT['text'], linespacing=1.3)

    ax_metrics.set_xlim(0, 1)
    ax_metrics.set_ylim(0, 1)
    ax_metrics.axis('off')

    plt.savefig(output_path / 'graphical_abstract_light.pdf', format='pdf',
                facecolor=COLORS_LIGHT['bg'], edgecolor='none')
    plt.savefig(output_path / 'graphical_abstract_light.png', format='png',
                facecolor=COLORS_LIGHT['bg'], edgecolor='none', dpi=400)
    plt.close()
    print(f"Saved light graphical abstract to {output_path}")


if __name__ == "__main__":
    results_path = Path('results/smb_final')
    output_path = results_path / 'figures_nature'
    output_path.mkdir(exist_ok=True)

    create_graphical_abstract(results_path, output_path)
    create_graphical_abstract_light(results_path, output_path)
