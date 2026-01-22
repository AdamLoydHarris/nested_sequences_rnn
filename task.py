"""
Task definition for nested sequence working memory study.

Implements three sequence types:
- Nested (ABABCDCD): Hierarchical chunk structure
- Flat repeat (ABCDABCD): Repetition without nesting
- Unique (EFGHIJKL): No repetition structure

Based on:
- Xie et al. (2022, Science) - Sequence geometry in PFC
- Liu et al. (2024, bioRxiv) - 2D neural geometry for hierarchical sequences
"""

import numpy as np
import torch
from enum import Enum
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


class SequenceType(Enum):
    NESTED = "nested"           # ABABCDCD - hierarchical chunks
    FLAT_REPEAT = "flat_repeat" # ABCDABCD - flat repetition
    UNIQUE = "unique"           # EFGHIJKL - no structure


@dataclass
class TrialInfo:
    """Metadata for a single trial."""
    seq_type: SequenceType
    sequence: List[int]
    items_used: List[int]
    local_ranks: List[int]      # Position within repeating unit (0 or 1)
    global_ranks: List[int]     # Chunk identity (0 or 1)
    repetition_ranks: List[int] # Which repetition (0 or 1)


class NestedSequenceTask:
    """
    Sequence working memory task with hierarchical structure.

    Trial structure:
    1. Fixation period
    2. Stimulus presentation (8 items sequentially)
    3. Delay period (maintenance)
    4. Response period (recall sequence)
    """

    def __init__(
        self,
        n_stimuli: int = 16,
        seq_length: int = 8,
        fixation_duration: int = 3,
        stimulus_duration: int = 2,
        delay_duration: int = 8,
        response_duration: int = 2,
    ):
        self.n_stimuli = n_stimuli
        self.seq_length = seq_length
        self.fixation_duration = fixation_duration
        self.stimulus_duration = stimulus_duration
        self.delay_duration = delay_duration
        self.response_duration = response_duration

        # Input: one-hot stimulus + fixation + go cue
        self.input_dim = n_stimuli + 2
        self.output_dim = n_stimuli + 1  # +1 for null response

        # Total trial length
        self.trial_length = (
            fixation_duration +
            seq_length * stimulus_duration +
            delay_duration +
            seq_length * response_duration
        )

    def generate_nested_sequence(self) -> Tuple[List[int], TrialInfo]:
        """
        Generate ABABCDCD sequence.

        Structure: Two chunks, each with 2 items repeated twice.
        - Chunk 1: positions 0-3 (items A, B repeated as ABAB)
        - Chunk 2: positions 4-7 (items C, D repeated as CDCD)
        """
        # Sample 4 distinct items
        items = np.random.choice(self.n_stimuli, size=4, replace=False).tolist()
        A, B, C, D = items

        sequence = [A, B, A, B, C, D, C, D]

        # Local rank: position within the 2-item unit (0=first, 1=second)
        local_ranks = [0, 1, 0, 1, 0, 1, 0, 1]

        # Global rank: chunk identity (0=chunk1, 1=chunk2)
        global_ranks = [0, 0, 0, 0, 1, 1, 1, 1]

        # Repetition rank: which repetition within chunk (0=first, 1=second)
        repetition_ranks = [0, 0, 1, 1, 0, 0, 1, 1]

        info = TrialInfo(
            seq_type=SequenceType.NESTED,
            sequence=sequence,
            items_used=items,
            local_ranks=local_ranks,
            global_ranks=global_ranks,
            repetition_ranks=repetition_ranks,
        )

        return sequence, info

    def generate_flat_repeat_sequence(self) -> Tuple[List[int], TrialInfo]:
        """
        Generate ABCDABCD sequence.

        Structure: 4 items repeated twice in same order.
        """
        items = np.random.choice(self.n_stimuli, size=4, replace=False).tolist()
        A, B, C, D = items

        sequence = [A, B, C, D, A, B, C, D]

        # Local rank: position within 4-item unit
        local_ranks = [0, 1, 2, 3, 0, 1, 2, 3]

        # Global rank: which repetition (0=first, 1=second)
        global_ranks = [0, 0, 0, 0, 1, 1, 1, 1]

        # Repetition rank: same as global for flat
        repetition_ranks = [0, 0, 0, 0, 1, 1, 1, 1]

        info = TrialInfo(
            seq_type=SequenceType.FLAT_REPEAT,
            sequence=sequence,
            items_used=items,
            local_ranks=local_ranks,
            global_ranks=global_ranks,
            repetition_ranks=repetition_ranks,
        )

        return sequence, info

    def generate_unique_sequence(self) -> Tuple[List[int], TrialInfo]:
        """
        Generate EFGHIJKL sequence.

        Structure: 8 unique items, no repetition.
        """
        items = np.random.choice(self.n_stimuli, size=8, replace=False).tolist()
        sequence = items.copy()

        # No meaningful local/global/repetition structure
        local_ranks = list(range(8))
        global_ranks = [0] * 8
        repetition_ranks = [0] * 8

        info = TrialInfo(
            seq_type=SequenceType.UNIQUE,
            sequence=sequence,
            items_used=items,
            local_ranks=local_ranks,
            global_ranks=global_ranks,
            repetition_ranks=repetition_ranks,
        )

        return sequence, info

    def generate_sequence(self, seq_type: SequenceType) -> Tuple[List[int], TrialInfo]:
        """Generate a sequence of the specified type."""
        if seq_type == SequenceType.NESTED:
            return self.generate_nested_sequence()
        elif seq_type == SequenceType.FLAT_REPEAT:
            return self.generate_flat_repeat_sequence()
        elif seq_type == SequenceType.UNIQUE:
            return self.generate_unique_sequence()
        else:
            raise ValueError(f"Unknown sequence type: {seq_type}")

    def generate_trial(self, seq_type: Optional[SequenceType] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, TrialInfo]:
        """
        Generate a single trial.

        Returns:
            inputs: (trial_length, input_dim)
            targets: (trial_length,) integer class labels
            mask: (trial_length,) loss mask (1 during response, 0 otherwise)
            info: Trial metadata
        """
        if seq_type is None:
            seq_type = np.random.choice(list(SequenceType))

        sequence, info = self.generate_sequence(seq_type)

        inputs = np.zeros((self.trial_length, self.input_dim), dtype=np.float32)
        targets = np.zeros(self.trial_length, dtype=np.int64)
        mask = np.zeros(self.trial_length, dtype=np.float32)

        t = 0

        # Fixation period
        inputs[t:t+self.fixation_duration, -2] = 1.0  # Fixation signal
        targets[t:t+self.fixation_duration] = self.n_stimuli  # Null response
        t += self.fixation_duration

        # Stimulus presentation
        for item in sequence:
            inputs[t:t+self.stimulus_duration, item] = 1.0  # Item input
            inputs[t:t+self.stimulus_duration, -2] = 1.0    # Maintain fixation
            targets[t:t+self.stimulus_duration] = self.n_stimuli  # Null response
            t += self.stimulus_duration

        # Delay period
        inputs[t:t+self.delay_duration, -2] = 1.0  # Maintain fixation
        targets[t:t+self.delay_duration] = self.n_stimuli  # Null response
        t += self.delay_duration

        # Response period
        for item in sequence:
            inputs[t:t+self.response_duration, -1] = 1.0  # Go cue
            targets[t:t+self.response_duration] = item
            mask[t:t+self.response_duration] = 1.0
            t += self.response_duration

        return inputs, targets, mask, info

    def generate_batch(
        self,
        batch_size: int,
        seq_type: Optional[SequenceType] = None,
        balanced: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[TrialInfo]]:
        """
        Generate a batch of trials.

        Args:
            batch_size: Number of trials
            seq_type: Type of sequence (random per trial if None)
            balanced: If True, equal numbers of each type

        Returns:
            inputs: (batch, trial_length, input_dim)
            targets: (batch, trial_length)
            mask: (batch, trial_length)
            infos: List of trial metadata
        """
        if balanced and seq_type is None:
            # Generate equal numbers of each type
            types = list(SequenceType)
            n_per_type = batch_size // len(types)
            seq_types = types * n_per_type
            # Fill remainder randomly
            remainder = batch_size - len(seq_types)
            seq_types.extend(np.random.choice(types, size=remainder).tolist())
            np.random.shuffle(seq_types)
        elif seq_type is not None:
            seq_types = [seq_type] * batch_size
        else:
            seq_types = [np.random.choice(list(SequenceType)) for _ in range(batch_size)]

        inputs_list = []
        targets_list = []
        mask_list = []
        infos = []

        for st in seq_types:
            inputs, targets, mask, info = self.generate_trial(st)
            inputs_list.append(torch.from_numpy(inputs))
            targets_list.append(torch.from_numpy(targets))
            mask_list.append(torch.from_numpy(mask))
            infos.append(info)

        inputs = torch.stack(inputs_list)
        targets = torch.stack(targets_list)
        mask = torch.stack(mask_list)

        return inputs, targets, mask, infos

    def info_to_dict(self, info: TrialInfo) -> Dict:
        """Convert TrialInfo to dictionary for JSON serialization."""
        return {
            'seq_type': info.seq_type.value,
            'sequence': info.sequence,
            'items_used': info.items_used,
            'local_ranks': info.local_ranks,
            'global_ranks': info.global_ranks,
            'repetition_ranks': info.repetition_ranks,
        }


if __name__ == "__main__":
    # Test task generation
    task = NestedSequenceTask()

    print(f"Task configuration:")
    print(f"  n_stimuli: {task.n_stimuli}")
    print(f"  seq_length: {task.seq_length}")
    print(f"  trial_length: {task.trial_length}")
    print(f"  input_dim: {task.input_dim}")
    print(f"  output_dim: {task.output_dim}")

    print("\nExample sequences:")
    for seq_type in SequenceType:
        seq, info = task.generate_sequence(seq_type)
        print(f"  {seq_type.value}: {seq}")
        print(f"    local_ranks: {info.local_ranks}")
        print(f"    global_ranks: {info.global_ranks}")
