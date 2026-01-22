"""
Comprehensive analysis suite for testing structured memory buffer (SMB),
spacetime attractor, and slot dynamics theories in RNNs trained on nested sequences.

Based on theoretical frameworks from:
- Xie et al. (2022, Science) - Subspace geometry of sequence memory
- El-Gaby et al. (2024, Nature) - Goal-progress cells, memory buffers
- Jensen et al. (2025, bioRxiv) - Spacetime attractor theory
- Whittington et al. (2024, Neuron) - Activity slots, EM-WM duality
- Liu et al. (2024, bioRxiv) - 2D neural geometry for hierarchical sequences

Analysis categories:
1. Subspace Geometry (Xie et al.)
2. Position/Item Decoding
3. Future Coding & Conveyor Belt (Jensen et al.)
4. Goal-Progress Tiling (El-Gaby et al.)
5. Slot Dynamics & Cross-Generalization (Whittington et al.)
6. Attractor Dynamics
7. Connectivity Analysis
8. Representational Similarity Analysis
"""

import numpy as np
from sklearn.linear_model import Ridge, RidgeClassifier, LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.linalg import svd, subspace_angles
from scipy.stats import pearsonr, spearmanr, ttest_ind
from scipy.spatial.distance import pdist, squareform
from typing import Dict, List, Tuple, Optional
import torch
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 1. SUBSPACE GEOMETRY ANALYSIS (Xie et al. 2022)
# =============================================================================

def xie_regression_subspace_analysis(
    hidden_states: np.ndarray,
    infos: List,
    n_stimuli: int = 16,
    n_pca_dims: int = 2,
    alpha: float = 1.0,
) -> Dict:
    """
    Xie et al. (2022) regression-based subspace analysis.

    Method:
    1. Create design matrix X: multi-hot vector (n_stimuli × n_positions)
    2. Ridge regression: h = X @ beta + epsilon for each neuron
    3. Split beta by position to get position-specific coding directions
    4. PCA on each position's betas to get 2D coding plane
    5. Compute principal angles between position planes

    Args:
        hidden_states: (n_trials, n_neurons) delay period activity
        infos: List of TrialInfo objects
        n_stimuli: Number of possible stimuli
        n_pca_dims: Dimensions for coding plane
        alpha: Ridge regularization

    Returns:
        Dict with subspace analysis results
    """
    n_trials, n_neurons = hidden_states.shape
    n_positions = 8

    # Build design matrix: (n_trials, n_stimuli * n_positions)
    X = np.zeros((n_trials, n_stimuli * n_positions))
    for i, info in enumerate(infos):
        seq = info.sequence if hasattr(info, 'sequence') else info['sequence']
        for pos, item in enumerate(seq):
            X[i, pos * n_stimuli + item] = 1

    # Ridge regression for each neuron
    # Beta shape: (n_stimuli * n_positions, n_neurons)
    reg = Ridge(alpha=alpha)
    reg.fit(X, hidden_states)
    beta = reg.coef_.T  # (n_stimuli * n_positions, n_neurons)

    # Split beta by position
    position_betas = {}
    for pos in range(n_positions):
        start_idx = pos * n_stimuli
        end_idx = (pos + 1) * n_stimuli
        position_betas[pos] = beta[start_idx:end_idx, :]  # (n_stimuli, n_neurons)

    # PCA on each position's betas to get coding plane
    position_planes = {}
    for pos in range(n_positions):
        betas = position_betas[pos]  # (n_stimuli, n_neurons)
        if betas.shape[0] >= n_pca_dims:
            pca = PCA(n_components=n_pca_dims)
            pca.fit(betas)
            # Coding plane: columns are principal directions in neuron space
            position_planes[pos] = pca.components_.T  # (n_neurons, n_pca_dims)

    # Compute principal angles between all pairs of position planes
    principal_angles = {}
    for i in range(n_positions):
        for j in range(i + 1, n_positions):
            if i in position_planes and j in position_planes:
                plane_i = position_planes[i]
                plane_j = position_planes[j]
                angles = np.degrees(subspace_angles(plane_i, plane_j))
                principal_angles[(i, j)] = angles.tolist()

    # Compute structural metrics for nested sequences
    # Within-chunk: positions 0-3 (chunk 1), 4-7 (chunk 2)
    within_chunk_angles = []
    between_chunk_angles = []

    for (i, j), angles in principal_angles.items():
        mean_angle = np.mean(angles)
        chunk_i = 0 if i < 4 else 1
        chunk_j = 0 if j < 4 else 1

        if chunk_i == chunk_j:
            within_chunk_angles.append(mean_angle)
        else:
            between_chunk_angles.append(mean_angle)

    # Same local rank vs different local rank
    same_local_rank_angles = []
    diff_local_rank_angles = []

    for (i, j), angles in principal_angles.items():
        mean_angle = np.mean(angles)
        local_i = i % 2
        local_j = j % 2

        if local_i == local_j:
            same_local_rank_angles.append(mean_angle)
        else:
            diff_local_rank_angles.append(mean_angle)

    return {
        'principal_angles': {str(k): v for k, v in principal_angles.items()},
        'mean_principal_angle': np.mean([np.mean(a) for a in principal_angles.values()]),
        'mean_within_chunk_angle': np.mean(within_chunk_angles) if within_chunk_angles else None,
        'mean_between_chunk_angle': np.mean(between_chunk_angles) if between_chunk_angles else None,
        'chunk_angle_difference': (np.mean(between_chunk_angles) - np.mean(within_chunk_angles)) if within_chunk_angles and between_chunk_angles else None,
        'mean_same_local_rank_angle': np.mean(same_local_rank_angles) if same_local_rank_angles else None,
        'mean_diff_local_rank_angle': np.mean(diff_local_rank_angles) if diff_local_rank_angles else None,
        'local_rank_angle_difference': (np.mean(diff_local_rank_angles) - np.mean(same_local_rank_angles)) if same_local_rank_angles and diff_local_rank_angles else None,
    }


# =============================================================================
# 2. POSITION/ITEM DECODING
# =============================================================================

def position_item_decoding(
    hidden_states: np.ndarray,
    infos: List,
    n_cv_folds: int = 5,
) -> Dict:
    """
    Decode position and item identity from neural activity.

    Tests:
    1. Position decoding accuracy
    2. Item decoding accuracy (at each position)
    3. Cross-position item decoding (train on one position, test on another)
    """
    n_trials = len(infos)

    # Extract sequences and build data arrays
    all_states = []
    all_positions = []
    all_items = []
    all_local_ranks = []
    all_global_ranks = []

    for trial_idx, info in enumerate(infos):
        seq = info.sequence if hasattr(info, 'sequence') else info['sequence']
        local_ranks = info.local_ranks if hasattr(info, 'local_ranks') else info.get('local_ranks', [0]*8)
        global_ranks = info.global_ranks if hasattr(info, 'global_ranks') else info.get('global_ranks', [0]*8)

        for pos in range(len(seq)):
            all_states.append(hidden_states[trial_idx])
            all_positions.append(pos)
            all_items.append(seq[pos])
            all_local_ranks.append(local_ranks[pos])
            all_global_ranks.append(global_ranks[pos])

    states = np.array(all_states)
    positions = np.array(all_positions)
    items = np.array(all_items)
    local_ranks = np.array(all_local_ranks)
    global_ranks = np.array(all_global_ranks)

    results = {}

    # Position decoding
    clf = RidgeClassifier(alpha=1.0)
    cv_scores = cross_val_score(clf, states, positions, cv=n_cv_folds)
    results['position_decoding'] = {
        'accuracy': float(np.mean(cv_scores)),
        'std': float(np.std(cv_scores)),
    }

    # Item decoding (overall)
    cv_scores = cross_val_score(clf, states, items, cv=n_cv_folds)
    results['item_decoding_overall'] = {
        'accuracy': float(np.mean(cv_scores)),
        'std': float(np.std(cv_scores)),
    }

    # Item decoding by position
    results['item_decoding_by_position'] = {}
    for pos in range(8):
        mask = positions == pos
        if mask.sum() > 20:
            pos_states = states[mask]
            pos_items = items[mask]
            if len(np.unique(pos_items)) > 1:
                cv_scores = cross_val_score(clf, pos_states, pos_items, cv=min(n_cv_folds, len(np.unique(pos_items))))
                results['item_decoding_by_position'][pos] = {
                    'accuracy': float(np.mean(cv_scores)),
                    'std': float(np.std(cv_scores)),
                }

    # Local rank decoding
    cv_scores = cross_val_score(clf, states, local_ranks, cv=n_cv_folds)
    results['local_rank_decoding'] = {
        'accuracy': float(np.mean(cv_scores)),
        'std': float(np.std(cv_scores)),
    }

    # Global rank (chunk) decoding
    cv_scores = cross_val_score(clf, states, global_ranks, cv=n_cv_folds)
    results['global_rank_decoding'] = {
        'accuracy': float(np.mean(cv_scores)),
        'std': float(np.std(cv_scores)),
    }

    # Cross-position item generalization
    # Train on positions 0-3, test on positions 4-7
    train_mask = positions < 4
    test_mask = positions >= 4

    if train_mask.sum() > 10 and test_mask.sum() > 10:
        clf.fit(states[train_mask], items[train_mask])
        cross_pos_acc = clf.score(states[test_mask], items[test_mask])
        results['cross_position_item_generalization'] = float(cross_pos_acc)

    return results


# =============================================================================
# 3. FUTURE CODING & CONVEYOR BELT (Jensen et al. 2025)
# =============================================================================

def future_coding_analysis(
    model,
    task,
    n_trials: int = 300,
    device: str = 'cpu',
) -> Dict:
    """
    Test if different neural subspaces encode different future positions.

    Following Jensen et al. (2025) spacetime attractor theory:
    1. Current activity should predict future sequence positions
    2. Different timepoints should have different predictive subspaces
    3. "Conveyor belt" structure: same decoder works across time shifts
    """
    from task import SequenceType

    model.eval()
    model = model.to(device)

    inputs, targets, _, infos = task.generate_batch(n_trials, SequenceType.NESTED)
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs, _ = model(inputs)
        hiddens = model.get_hidden_states().cpu().numpy()

    n_batch, n_time, n_hidden = hiddens.shape

    # Get stimulus presentation timepoints
    stim_start = task.fixation_duration
    stim_times = [stim_start + i * task.stimulus_duration + task.stimulus_duration // 2
                  for i in range(task.seq_length)]

    # Extract sequences
    sequences = np.array([info.sequence if hasattr(info, 'sequence') else info['sequence'] for info in infos])

    results = {
        'current_item_decoding': [],
        'next_item_decoding': [],
        'future2_item_decoding': [],
        'conveyor_belt_transfer': [],
        'chunk_boundary_effect': {},
    }

    clf = RidgeClassifier(alpha=1.0)
    n_train = int(0.7 * n_trials)

    # Test future coding at each position
    for t_idx in range(len(stim_times) - 2):
        t = stim_times[t_idx]
        H = hiddens[:, t, :]

        current_item = sequences[:, t_idx]
        next_item = sequences[:, t_idx + 1]
        future2_item = sequences[:, t_idx + 2]

        # Current item decoding
        clf.fit(H[:n_train], current_item[:n_train])
        results['current_item_decoding'].append(float(clf.score(H[n_train:], current_item[n_train:])))

        # Next item decoding (future coding)
        clf.fit(H[:n_train], next_item[:n_train])
        results['next_item_decoding'].append(float(clf.score(H[n_train:], next_item[n_train:])))

        # Future+2 item decoding
        clf.fit(H[:n_train], future2_item[:n_train])
        results['future2_item_decoding'].append(float(clf.score(H[n_train:], future2_item[n_train:])))

    # Conveyor belt: train at time t for next item, test at time t+1 for its next item
    for t_idx in range(len(stim_times) - 3):
        t1 = stim_times[t_idx]
        t2 = stim_times[t_idx + 1]

        H1 = hiddens[:, t1, :]
        H2 = hiddens[:, t2, :]

        item_at_t1_plus1 = sequences[:, t_idx + 1]
        item_at_t2_plus1 = sequences[:, t_idx + 2]

        # Train on H1 -> item at t1+1
        clf.fit(H1[:n_train], item_at_t1_plus1[:n_train])

        # Test same decoder on H2 -> item at t2+1
        conveyor_acc = clf.score(H2[n_train:], item_at_t2_plus1[n_train:])
        results['conveyor_belt_transfer'].append(float(conveyor_acc))

    # Chunk boundary effect: compare future coding before/after boundary
    # Boundary is between positions 3 and 4
    pre_boundary = results['next_item_decoding'][2] if len(results['next_item_decoding']) > 2 else None  # pos 2->3
    at_boundary = results['next_item_decoding'][3] if len(results['next_item_decoding']) > 3 else None   # pos 3->4
    post_boundary = results['next_item_decoding'][4] if len(results['next_item_decoding']) > 4 else None # pos 4->5

    results['chunk_boundary_effect'] = {
        'pre_boundary_accuracy': pre_boundary,
        'at_boundary_accuracy': at_boundary,
        'post_boundary_accuracy': post_boundary,
        'boundary_drop': (pre_boundary - at_boundary) if pre_boundary and at_boundary else None,
    }

    # Summary statistics
    results['mean_current_decoding'] = float(np.mean(results['current_item_decoding']))
    results['mean_next_decoding'] = float(np.mean(results['next_item_decoding']))
    results['mean_future2_decoding'] = float(np.mean(results['future2_item_decoding']))
    results['mean_conveyor_belt'] = float(np.mean(results['conveyor_belt_transfer'])) if results['conveyor_belt_transfer'] else 0

    return results


# =============================================================================
# 4. GOAL-PROGRESS TILING (El-Gaby et al. 2024)
# =============================================================================

def goal_progress_tiling_analysis(
    model,
    task,
    n_trials: int = 300,
    device: str = 'cpu',
) -> Dict:
    """
    Test if neurons tile progress through the sequence like goal-progress cells.

    Following El-Gaby et al. (2024):
    1. Individual neurons have preferred positions (like place cells)
    2. Neurons tile the sequence space smoothly
    3. Tuning is modulated by chunk structure
    4. Neurons stretch/compress tuning for different contexts
    """
    from task import SequenceType

    model.eval()
    model = model.to(device)

    inputs, targets, _, infos = task.generate_batch(n_trials, SequenceType.NESTED)
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs, _ = model(inputs)
        hiddens = model.get_hidden_states().cpu().numpy()

    n_trials_actual, n_time, n_hidden = hiddens.shape

    # Get activity at each stimulus presentation
    stim_start = task.fixation_duration
    position_activities = []

    for pos in range(task.seq_length):
        t = stim_start + pos * task.stimulus_duration + task.stimulus_duration // 2
        position_activities.append(hiddens[:, t, :])

    position_activities = np.array(position_activities)  # (8, n_trials, n_hidden)
    mean_by_position = position_activities.mean(axis=1)  # (8, n_hidden)

    # Find preferred position for each neuron
    preferred_positions = np.argmax(np.abs(mean_by_position), axis=0)

    # Compute position selectivity
    max_response = np.max(np.abs(mean_by_position), axis=0)
    mean_response = np.mean(np.abs(mean_by_position), axis=0)
    selectivity = (max_response - mean_response) / (max_response + 1e-10)

    # Tuning curve smoothness (correlation between adjacent positions)
    smoothness_scores = []
    for neuron in range(n_hidden):
        curve = mean_by_position[:, neuron]
        adjacent_corrs = []
        for i in range(7):
            if curve[i] != 0 or curve[i+1] != 0:
                corr = np.corrcoef(
                    position_activities[i, :, neuron],
                    position_activities[i+1, :, neuron]
                )[0, 1]
                if not np.isnan(corr):
                    adjacent_corrs.append(corr)
        if adjacent_corrs:
            smoothness_scores.append(np.mean(adjacent_corrs))

    # Chunk-modulated tuning
    chunk1_positions = [0, 1, 2, 3]
    chunk2_positions = [4, 5, 6, 7]

    within_chunk_corrs = []
    between_chunk_corrs = []

    for neuron in range(min(n_hidden, 100)):  # Sample neurons for efficiency
        # Within chunk correlations
        for i in chunk1_positions:
            for j in chunk1_positions:
                if i < j:
                    act_i = position_activities[i, :, neuron]
                    act_j = position_activities[j, :, neuron]
                    if act_i.std() > 0 and act_j.std() > 0:
                        r, _ = pearsonr(act_i, act_j)
                        if not np.isnan(r):
                            within_chunk_corrs.append(r)

        # Between chunk correlations
        for i in chunk1_positions:
            for j in chunk2_positions:
                act_i = position_activities[i, :, neuron]
                act_j = position_activities[j, :, neuron]
                if act_i.std() > 0 and act_j.std() > 0:
                    r, _ = pearsonr(act_i, act_j)
                    if not np.isnan(r):
                        between_chunk_corrs.append(r)

    # Position coverage uniformity
    position_counts = np.bincount(preferred_positions, minlength=8)
    coverage_uniformity = 1 - (np.std(position_counts) / (np.mean(position_counts) + 1e-10))

    return {
        'preferred_position_distribution': position_counts.tolist(),
        'mean_selectivity': float(np.mean(selectivity)),
        'n_highly_selective': int(np.sum(selectivity > 0.3)),
        'frac_highly_selective': float(np.mean(selectivity > 0.3)),
        'mean_tuning_smoothness': float(np.mean(smoothness_scores)) if smoothness_scores else 0,
        'coverage_uniformity': float(coverage_uniformity),
        'mean_within_chunk_correlation': float(np.mean(within_chunk_corrs)) if within_chunk_corrs else 0,
        'mean_between_chunk_correlation': float(np.mean(between_chunk_corrs)) if between_chunk_corrs else 0,
        'chunk_modulation': float(np.mean(within_chunk_corrs) - np.mean(between_chunk_corrs)) if within_chunk_corrs and between_chunk_corrs else 0,
    }


# =============================================================================
# 5. SLOT DYNAMICS & CROSS-GENERALIZATION (Whittington et al. 2024)
# =============================================================================

def slot_dynamics_analysis(
    hidden_states: np.ndarray,
    infos: List,
) -> Dict:
    """
    Test for activity slot properties following Whittington et al. (2024).

    Tests:
    1. Position invariance: Does position decoding generalize across items?
    2. Content-structure separation: Are item and position in separable subspaces?
    3. Controllable subspaces: Can position/item be independently manipulated?
    """
    n_trials = len(infos)

    # Build arrays
    all_states = []
    all_positions = []
    all_items = []

    for trial_idx, info in enumerate(infos):
        seq = info.sequence if hasattr(info, 'sequence') else info['sequence']
        for pos in range(len(seq)):
            all_states.append(hidden_states[trial_idx])
            all_positions.append(pos)
            all_items.append(seq[pos])

    states = np.array(all_states)
    positions = np.array(all_positions)
    items = np.array(all_items)

    results = {}

    # 1. Position invariance: Leave-one-item-out cross-validation
    clf = RidgeClassifier(alpha=1.0)

    # Standard CV
    cv_scores = cross_val_score(clf, states, positions, cv=5)
    results['position_cv_accuracy'] = float(np.mean(cv_scores))

    # Leave-item-out
    unique_items = np.unique(items)
    logo_scores = []
    for held_out_item in unique_items[:10]:  # Sample for efficiency
        train_mask = items != held_out_item
        test_mask = items == held_out_item
        if test_mask.sum() >= 5:
            clf.fit(states[train_mask], positions[train_mask])
            logo_scores.append(clf.score(states[test_mask], positions[test_mask]))

    results['position_leave_item_out_accuracy'] = float(np.mean(logo_scores)) if logo_scores else 0
    results['position_generalization_gap'] = results['position_cv_accuracy'] - results['position_leave_item_out_accuracy']

    # 2. Content-structure separation via variance partitioning
    n_neurons = states.shape[1]

    # Compute R² for position, item, and their combination
    from sklearn.preprocessing import OneHotEncoder

    pos_encoder = OneHotEncoder(sparse=False)
    pos_onehot = pos_encoder.fit_transform(positions.reshape(-1, 1))

    item_encoder = OneHotEncoder(sparse=False)
    item_onehot = item_encoder.fit_transform(items.reshape(-1, 1))

    combined = np.hstack([pos_onehot, item_onehot])

    pos_r2_list = []
    item_r2_list = []
    combined_r2_list = []

    reg = Ridge(alpha=1.0)
    for neuron in range(min(n_neurons, 50)):
        y = states[:, neuron]
        if y.std() > 0:
            # Position only
            reg.fit(pos_onehot, y)
            ss_res = np.sum((y - reg.predict(pos_onehot))**2)
            ss_tot = np.sum((y - y.mean())**2)
            pos_r2_list.append(1 - ss_res/ss_tot)

            # Item only
            reg.fit(item_onehot, y)
            ss_res = np.sum((y - reg.predict(item_onehot))**2)
            item_r2_list.append(1 - ss_res/ss_tot)

            # Combined
            reg.fit(combined, y)
            ss_res = np.sum((y - reg.predict(combined))**2)
            combined_r2_list.append(1 - ss_res/ss_tot)

    results['mean_position_r2'] = float(np.mean(pos_r2_list)) if pos_r2_list else 0
    results['mean_item_r2'] = float(np.mean(item_r2_list)) if item_r2_list else 0
    results['mean_combined_r2'] = float(np.mean(combined_r2_list)) if combined_r2_list else 0
    results['interaction_variance'] = results['mean_combined_r2'] - results['mean_position_r2'] - results['mean_item_r2']

    # 3. Subspace orthogonality between position and item coding
    pos_reg = Ridge(alpha=1.0).fit(pos_onehot, states)
    item_reg = Ridge(alpha=1.0).fit(item_onehot, states)

    pos_weights = pos_reg.coef_  # (n_neurons, n_positions)
    item_weights = item_reg.coef_  # (n_neurons, n_items)

    # PCA on weights (transpose so positions/items are samples, neurons are features)
    n_dims = min(5, pos_weights.T.shape[0], item_weights.T.shape[0])
    pos_pca = PCA(n_components=n_dims).fit(pos_weights.T)  # (n_positions, n_neurons)
    item_pca = PCA(n_components=n_dims).fit(item_weights.T)  # (n_items, n_neurons)

    # Principal angles
    angles = np.degrees(subspace_angles(pos_pca.components_.T, item_pca.components_.T))
    results['position_item_subspace_angles'] = angles.tolist()
    results['mean_position_item_angle'] = float(np.mean(angles))
    results['subspace_orthogonality'] = float(np.mean(angles) / 90)

    return results


def cross_sequence_generalization(
    model,
    task,
    n_trials: int = 200,
    device: str = 'cpu',
) -> Dict:
    """
    Test if structural representations generalize across sequence types.

    Train position decoder on one sequence type, test on another.
    High transfer suggests shared structural scaffold.
    """
    from task import SequenceType

    model.eval()
    model = model.to(device)

    # Collect data for each sequence type
    data_by_type = {}

    for seq_type in [SequenceType.NESTED, SequenceType.FLAT_REPEAT, SequenceType.UNIQUE]:
        inputs, targets, _, infos = task.generate_batch(n_trials, seq_type)
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs, _ = model(inputs)
            hiddens = model.get_hidden_states().cpu().numpy()

        delay_start = task.fixation_duration + task.seq_length * task.stimulus_duration
        delay_idx = delay_start + task.delay_duration // 2
        delay_states = hiddens[:, delay_idx, :]

        # Expand by position
        all_states = []
        all_positions = []
        for trial_idx in range(n_trials):
            for pos in range(8):
                all_states.append(delay_states[trial_idx])
                all_positions.append(pos)

        data_by_type[seq_type.value] = {
            'states': np.array(all_states),
            'positions': np.array(all_positions),
        }

    # Cross-type generalization matrix
    results = {'transfer_matrix': {}}
    clf = RidgeClassifier(alpha=1.0)

    for train_type in data_by_type:
        results['transfer_matrix'][train_type] = {}
        train_data = data_by_type[train_type]

        # Within-type accuracy (train/test split)
        n = len(train_data['positions'])
        n_train = n // 2
        clf.fit(train_data['states'][:n_train], train_data['positions'][:n_train])
        within_acc = clf.score(train_data['states'][n_train:], train_data['positions'][n_train:])

        for test_type in data_by_type:
            test_data = data_by_type[test_type]
            clf.fit(train_data['states'], train_data['positions'])
            acc = clf.score(test_data['states'], test_data['positions'])
            results['transfer_matrix'][train_type][test_type] = float(acc)

    # Summary metrics
    nested_to_flat = results['transfer_matrix']['nested']['flat_repeat']
    flat_to_nested = results['transfer_matrix']['flat_repeat']['nested']
    nested_within = results['transfer_matrix']['nested']['nested']

    results['nested_flat_transfer'] = float((nested_to_flat + flat_to_nested) / 2)
    results['structure_specificity'] = float(nested_within - nested_to_flat) if nested_within and nested_to_flat else 0

    return results


# =============================================================================
# 6. ATTRACTOR DYNAMICS
# =============================================================================

def attractor_dynamics_analysis(
    model,
    task,
    n_trials: int = 200,
    device: str = 'cpu',
) -> Dict:
    """
    Test for attractor-like dynamics during delay period.

    Following Jensen et al., tests:
    1. Stability: Low variance during delay
    2. Distinctiveness: High variance across trials
    3. Discrete states: Clustering of delay activity
    """
    from task import SequenceType
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    model.eval()
    model = model.to(device)

    results = {}

    for seq_type in SequenceType:
        inputs, targets, _, infos = task.generate_batch(n_trials, seq_type)
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs, _ = model(inputs)
            hiddens = model.get_hidden_states().cpu().numpy()

        delay_start = task.fixation_duration + task.seq_length * task.stimulus_duration
        delay_end = delay_start + task.delay_duration
        delay_activity = hiddens[:, delay_start:delay_end, :]

        # Temporal stability (variance across time)
        temporal_var = np.var(delay_activity, axis=1).mean()

        # Trial distinctiveness (variance across trials)
        delay_mid = delay_start + task.delay_duration // 2
        trial_var = np.var(hiddens[:, delay_mid, :], axis=0).mean()

        # Trajectory analysis
        path_lengths = []
        euclidean_distances = []
        for trial in range(n_trials):
            trajectory = delay_activity[trial]
            diffs = np.diff(trajectory, axis=0)
            path_length = np.sum(np.linalg.norm(diffs, axis=1))
            euclidean = np.linalg.norm(trajectory[-1] - trajectory[0])
            path_lengths.append(path_length)
            euclidean_distances.append(euclidean)

        straightness = np.mean(path_lengths) / (np.mean(euclidean_distances) + 1e-10)

        # Clustering (do delay states form discrete attractors?)
        delay_states = hiddens[:, delay_mid, :]
        if len(np.unique([info.sequence if hasattr(info, 'sequence') else info['sequence'] for info in infos], axis=0)) > 1:
            try:
                kmeans = KMeans(n_clusters=min(8, n_trials // 10), random_state=42, n_init=10)
                labels = kmeans.fit_predict(delay_states)
                silhouette = silhouette_score(delay_states, labels)
            except:
                silhouette = 0
        else:
            silhouette = 0

        results[seq_type.value] = {
            'temporal_variance': float(temporal_var),
            'trial_variance': float(trial_var),
            'mean_path_length': float(np.mean(path_lengths)),
            'mean_euclidean_distance': float(np.mean(euclidean_distances)),
            'trajectory_straightness': float(straightness),
            'cluster_silhouette': float(silhouette),
        }

    return results


# =============================================================================
# 7. CONNECTIVITY ANALYSIS
# =============================================================================

def connectivity_analysis(model) -> Dict:
    """
    Analyze structure in recurrent connectivity.

    Tests whether task structure is embedded in weights.
    """
    model.eval()

    # Extract recurrent weights
    if hasattr(model, 'gru'):
        W_hh = model.gru.weight_hh_l0.detach().cpu().numpy()
        hidden_size = W_hh.shape[1]
        # GRU has 3 gates; use the candidate gate (most relevant for dynamics)
        W = W_hh[2*hidden_size:, :]
    elif hasattr(model, 'lstm'):
        W_hh = model.lstm.weight_hh_l0.detach().cpu().numpy()
        hidden_size = W_hh.shape[1]
        # LSTM has 4 gates; use the cell gate
        W = W_hh[2*hidden_size:3*hidden_size, :]
    else:
        return {'error': 'Unknown model type'}

    # Eigenspectrum
    eigenvalues = np.linalg.eigvals(W)
    spectral_radius = np.max(np.abs(eigenvalues))

    # Singular value decomposition
    U, s, Vt = np.linalg.svd(W)

    # Effective rank
    s_norm = s / s.sum()
    effective_rank = np.exp(-np.sum(s_norm * np.log(s_norm + 1e-10)))

    # Low-rank structure
    rank_1_approx = s[0] * np.outer(U[:, 0], Vt[0, :])
    rank_1_error = np.linalg.norm(W - rank_1_approx, 'fro') / np.linalg.norm(W, 'fro')

    # Weight statistics
    weight_norm = np.linalg.norm(W, 'fro')
    weight_sparsity = np.mean(np.abs(W) < 0.01)

    return {
        'spectral_radius': float(spectral_radius),
        'effective_rank': float(effective_rank),
        'rank_1_error': float(rank_1_error),
        'top_singular_values': s[:10].tolist(),
        'singular_value_ratio': float(s[0] / s[-1]) if s[-1] > 0 else float('inf'),
        'weight_norm': float(weight_norm),
        'weight_sparsity': float(weight_sparsity),
        'n_positive_eigenvalues': int(np.sum(np.real(eigenvalues) > 0)),
        'n_negative_eigenvalues': int(np.sum(np.real(eigenvalues) < 0)),
    }


# =============================================================================
# 8. REPRESENTATIONAL SIMILARITY ANALYSIS
# =============================================================================

def representational_similarity_analysis(
    hidden_states: np.ndarray,
    infos: List,
) -> Dict:
    """
    Compute representational dissimilarity matrices (RDMs) and test
    for structural organization.
    """
    n_trials = len(infos)

    # Compute mean activity for each sequence
    sequences = [tuple(info.sequence if hasattr(info, 'sequence') else info['sequence']) for info in infos]
    unique_sequences = list(set(sequences))

    if len(unique_sequences) < 3:
        return {'error': 'Not enough unique sequences for RSA'}

    # Compute mean activity for each unique sequence
    seq_means = {}
    for seq in unique_sequences:
        mask = [s == seq for s in sequences]
        if sum(mask) > 0:
            seq_means[seq] = hidden_states[mask].mean(axis=0)

    # Build RDM (correlation distance)
    n_seq = len(seq_means)
    rdm = np.zeros((n_seq, n_seq))
    seq_list = list(seq_means.keys())

    for i in range(n_seq):
        for j in range(n_seq):
            if i != j:
                corr, _ = pearsonr(seq_means[seq_list[i]], seq_means[seq_list[j]])
                rdm[i, j] = 1 - corr

    # Compute model RDMs for different hypotheses
    # 1. Item identity model: sequences sharing items are similar
    item_rdm = np.zeros((n_seq, n_seq))
    for i in range(n_seq):
        for j in range(n_seq):
            if i != j:
                shared = len(set(seq_list[i]) & set(seq_list[j]))
                # Normalize by max possible shared items
                max_shared = max(len(set(seq_list[i])), len(set(seq_list[j])))
                item_rdm[i, j] = 1 - shared / max_shared if max_shared > 0 else 0

    # 2. Position model: sequences with same structure are similar
    # (For nested: similar chunk patterns)

    # Compute correlation with model RDMs
    rdm_flat = squareform(rdm)
    item_rdm_flat = squareform(item_rdm)

    if len(rdm_flat) > 0 and len(item_rdm_flat) > 0:
        item_corr, _ = spearmanr(rdm_flat, item_rdm_flat)
    else:
        item_corr = 0

    return {
        'n_unique_sequences': n_seq,
        'mean_rdm_distance': float(np.mean(rdm[np.triu_indices(n_seq, k=1)])),
        'item_model_correlation': float(item_corr) if not np.isnan(item_corr) else 0,
    }


# =============================================================================
# MAIN ANALYSIS RUNNER
# =============================================================================

def run_comprehensive_analysis(
    model,
    task,
    n_trials: int = 500,
    device: str = 'cpu',
) -> Dict:
    """
    Run all analyses and return comprehensive results.
    """
    from task import SequenceType

    model.eval()
    model = model.to(device)

    print("Running comprehensive SMB analysis suite...")
    results = {}

    # Generate data for each sequence type
    print("  Generating trial data...")
    data_by_type = {}
    for seq_type in SequenceType:
        inputs, targets, _, infos = task.generate_batch(n_trials, seq_type)
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs, _ = model(inputs)
            hiddens = model.get_hidden_states().cpu().numpy()

        delay_start = task.fixation_duration + task.seq_length * task.stimulus_duration
        delay_idx = delay_start + task.delay_duration // 2
        delay_states = hiddens[:, delay_idx, :]

        data_by_type[seq_type.value] = {
            'hidden_states': delay_states,
            'infos': [task.info_to_dict(info) if hasattr(task, 'info_to_dict') else
                     {'sequence': info.sequence, 'local_ranks': info.local_ranks,
                      'global_ranks': info.global_ranks, 'repetition_ranks': info.repetition_ranks}
                     for info in infos],
        }

    # 1. Xie et al. subspace analysis
    print("  1. Running Xie et al. subspace analysis...")
    results['subspace_geometry'] = {}
    for seq_type in SequenceType:
        results['subspace_geometry'][seq_type.value] = xie_regression_subspace_analysis(
            data_by_type[seq_type.value]['hidden_states'],
            data_by_type[seq_type.value]['infos'],
        )

    # 2. Position/item decoding
    print("  2. Running position/item decoding...")
    results['decoding'] = {}
    for seq_type in SequenceType:
        results['decoding'][seq_type.value] = position_item_decoding(
            data_by_type[seq_type.value]['hidden_states'],
            data_by_type[seq_type.value]['infos'],
        )

    # 3. Future coding (Jensen et al.)
    print("  3. Running future coding analysis...")
    results['future_coding'] = future_coding_analysis(model, task, n_trials=n_trials, device=device)

    # 4. Goal-progress tiling (El-Gaby et al.)
    print("  4. Running goal-progress tiling analysis...")
    results['goal_progress'] = goal_progress_tiling_analysis(model, task, n_trials=n_trials, device=device)

    # 5. Slot dynamics (Whittington et al.)
    print("  5. Running slot dynamics analysis...")
    results['slot_dynamics'] = {}
    for seq_type in SequenceType:
        results['slot_dynamics'][seq_type.value] = slot_dynamics_analysis(
            data_by_type[seq_type.value]['hidden_states'],
            data_by_type[seq_type.value]['infos'],
        )

    # 6. Cross-sequence generalization
    print("  6. Running cross-sequence generalization...")
    results['cross_generalization'] = cross_sequence_generalization(model, task, n_trials=n_trials//2, device=device)

    # 7. Attractor dynamics
    print("  7. Running attractor dynamics analysis...")
    results['attractor_dynamics'] = attractor_dynamics_analysis(model, task, n_trials=n_trials//2, device=device)

    # 8. Connectivity analysis
    print("  8. Running connectivity analysis...")
    results['connectivity'] = connectivity_analysis(model)

    # 9. RSA
    print("  9. Running representational similarity analysis...")
    results['rsa'] = {}
    for seq_type in SequenceType:
        results['rsa'][seq_type.value] = representational_similarity_analysis(
            data_by_type[seq_type.value]['hidden_states'],
            data_by_type[seq_type.value]['infos'],
        )

    print("Analysis complete!")
    return results


if __name__ == "__main__":
    import sys
    import json
    from pathlib import Path

    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    results_path = Path(results_dir)

    from task import NestedSequenceTask
    from train import load_model

    task = NestedSequenceTask()

    model_path = results_path / "final_model.pt"
    if not model_path.exists():
        model_path = results_path / "best_model.pt"

    if model_path.exists():
        model, checkpoint = load_model(str(model_path), task)
        device = 'cpu'

        results = run_comprehensive_analysis(model, task, device=device)

        # Save results
        output_path = results_path / "comprehensive_analysis.json"

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

        with open(output_path, 'w') as f:
            json.dump(convert(results), f, indent=2)

        print(f"\nResults saved to {output_path}")
    else:
        print(f"Model not found at {model_path}")
