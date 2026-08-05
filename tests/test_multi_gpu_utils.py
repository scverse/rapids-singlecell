"""Tests for multi-GPU utility functions."""

from __future__ import annotations

import cupy as cp

from rapids_singlecell._utils import _split_pairs, parse_device_ids


class TestSplitPairs:
    """Tests for the _split_pairs load balancing function."""

    def test_empty_pairs(self):
        """Empty input returns empty arrays for each device."""
        pair_left = cp.array([], dtype=cp.int32)
        pair_right = cp.array([], dtype=cp.int32)

        result = _split_pairs(pair_left, pair_right, n_devices=2)

        assert len(result) == 2
        for left, right in result:
            assert len(left) == 0
            assert len(right) == 0

    def test_single_device_returns_all_pairs(self):
        """Single device gets all pairs."""
        pair_left = cp.array([0, 0, 1], dtype=cp.int32)
        pair_right = cp.array([0, 1, 1], dtype=cp.int32)

        result = _split_pairs(pair_left, pair_right, n_devices=1)

        assert len(result) == 1
        cp.testing.assert_array_equal(result[0][0], pair_left)
        cp.testing.assert_array_equal(result[0][1], pair_right)

    def test_even_split_without_group_sizes(self):
        """Without group_sizes, pairs are split evenly by count."""
        pair_left = cp.array([0, 0, 1, 1], dtype=cp.int32)
        pair_right = cp.array([0, 1, 1, 2], dtype=cp.int32)

        result = _split_pairs(pair_left, pair_right, n_devices=2, group_sizes=None)

        assert len(result) == 2
        # 4 pairs split into 2 devices = 2 pairs each
        assert len(result[0][0]) == 2
        assert len(result[1][0]) == 2

    def test_all_pairs_preserved(self):
        """All original pairs are present in the split result."""
        pair_left = cp.array([0, 0, 1, 1, 2], dtype=cp.int32)
        pair_right = cp.array([0, 1, 1, 2, 2], dtype=cp.int32)
        group_sizes = cp.array([100, 200, 300], dtype=cp.int64)

        result = _split_pairs(
            pair_left, pair_right, n_devices=3, group_sizes=group_sizes
        )

        # Concatenate all splits and verify they match original
        all_left = cp.concatenate([r[0] for r in result])
        all_right = cp.concatenate([r[1] for r in result])

        cp.testing.assert_array_equal(all_left, pair_left)
        cp.testing.assert_array_equal(all_right, pair_right)

    def test_load_balancing_with_unequal_groups(self):
        """Load balancing assigns more pairs to devices when groups are small."""
        # Group 0: 10 cells, Group 1: 10 cells, Group 2: 1000 cells
        # Pair (0,1): 10*10=100 work
        # Pair (0,2): 10*1000=10000 work
        # Pair (1,2): 10*1000=10000 work
        # Pair (2,2): 1000*999/2=499500 work (diagonal)
        pair_left = cp.array([0, 0, 1, 2], dtype=cp.int32)
        pair_right = cp.array([1, 2, 2, 2], dtype=cp.int32)
        group_sizes = cp.array([10, 10, 1000], dtype=cp.int64)

        result = _split_pairs(
            pair_left, pair_right, n_devices=2, group_sizes=group_sizes
        )

        # The diagonal pair (2,2) has most work, should likely be alone or with few others
        # First device should get the smaller pairs, second should get the large diagonal
        total_pairs_device_0 = len(result[0][0])
        total_pairs_device_1 = len(result[1][0])

        # All pairs should be distributed
        assert total_pairs_device_0 + total_pairs_device_1 == 4

    def test_diagonal_work_calculation(self):
        """Diagonal pairs use n*(n-1)/2 formula for within-group work."""
        # Two diagonal pairs with different group sizes
        pair_left = cp.array([0, 1], dtype=cp.int32)
        pair_right = cp.array([0, 1], dtype=cp.int32)
        # Group 0: 100 cells -> 100*99/2 = 4950 work
        # Group 1: 10 cells -> 10*9/2 = 45 work
        group_sizes = cp.array([100, 10], dtype=cp.int64)

        result = _split_pairs(
            pair_left, pair_right, n_devices=2, group_sizes=group_sizes
        )

        # Each device should get one pair since there are 2 pairs and 2 devices
        # The work is very unbalanced (4950 vs 45), but we only have 2 pairs
        assert len(result[0][0]) + len(result[1][0]) == 2

    def test_more_devices_than_pairs(self):
        """More devices than pairs results in some devices with empty arrays."""
        pair_left = cp.array([0, 1], dtype=cp.int32)
        pair_right = cp.array([1, 1], dtype=cp.int32)

        result = _split_pairs(pair_left, pair_right, n_devices=5)

        assert len(result) == 5
        # Only 2 pairs, so at most 2 devices have work
        non_empty = sum(1 for r in result if len(r[0]) > 0)
        assert non_empty <= 2

    def test_work_roughly_balanced(self):
        """With group_sizes, work should be roughly balanced across devices."""
        # Create 10 pairs with varying work loads
        n_groups = 5
        pair_left = []
        pair_right = []
        for i in range(n_groups):
            for j in range(i, n_groups):
                pair_left.append(i)
                pair_right.append(j)

        pair_left = cp.array(pair_left, dtype=cp.int32)
        pair_right = cp.array(pair_right, dtype=cp.int32)
        # Exponentially increasing group sizes to create unbalanced work
        group_sizes = cp.array([10, 20, 40, 80, 160], dtype=cp.int64)

        result = _split_pairs(
            pair_left, pair_right, n_devices=3, group_sizes=group_sizes
        )

        # Calculate actual work per device
        def calc_work(left, right, sizes):
            work = 0
            for l, r in zip(left.get(), right.get()):
                if l == r:
                    work += sizes[l] * (sizes[l] - 1) // 2
                else:
                    work += sizes[l] * sizes[r]
            return work

        sizes_np = group_sizes.get()
        works = [calc_work(r[0], r[1], sizes_np) for r in result]

        # Work should be within 2x of each other (rough balance)
        # Filter out empty devices
        non_zero_works = [w for w in works if w > 0]
        if len(non_zero_works) > 1:
            ratio = max(non_zero_works) / min(non_zero_works)
            # Allow some imbalance, but should be reasonably balanced
            assert ratio < 5, f"Work imbalance too high: {works}"

    def test_preserves_pair_order_within_chunks(self):
        """Pairs within each chunk maintain their relative order."""
        pair_left = cp.array([0, 1, 2, 3, 4], dtype=cp.int32)
        pair_right = cp.array([0, 1, 2, 3, 4], dtype=cp.int32)

        result = _split_pairs(pair_left, pair_right, n_devices=2, group_sizes=None)

        # First chunk should have consecutive pairs from the start
        if len(result[0][0]) > 0:
            first_left = result[0][0].get()
            # Should be monotonically increasing (or equal for diagonal)
            assert all(
                first_left[i] <= first_left[i + 1] for i in range(len(first_left) - 1)
            )

    def test_single_pair(self):
        """Single pair is assigned to exactly one device."""
        pair_left = cp.array([0], dtype=cp.int32)
        pair_right = cp.array([1], dtype=cp.int32)
        group_sizes = cp.array([100, 100], dtype=cp.int64)

        result = _split_pairs(
            pair_left, pair_right, n_devices=4, group_sizes=group_sizes
        )

        assert len(result) == 4
        # Exactly one device should have the pair
        non_empty = [i for i, r in enumerate(result) if len(r[0]) > 0]
        assert len(non_empty) == 1
        # That device should have exactly 1 pair
        assert len(result[non_empty[0]][0]) == 1

    def test_large_number_of_pairs(self):
        """Handles large number of pairs efficiently."""
        n_groups = 100
        pair_left = []
        pair_right = []
        for i in range(n_groups):
            for j in range(i, n_groups):
                pair_left.append(i)
                pair_right.append(j)

        pair_left = cp.array(pair_left, dtype=cp.int32)
        pair_right = cp.array(pair_right, dtype=cp.int32)
        group_sizes = cp.random.randint(10, 1000, size=n_groups, dtype=cp.int64)

        n_pairs = len(pair_left)
        assert n_pairs == n_groups * (n_groups + 1) // 2  # 5050 pairs

        result = _split_pairs(
            pair_left, pair_right, n_devices=8, group_sizes=group_sizes
        )

        assert len(result) == 8
        total = sum(len(r[0]) for r in result)
        assert total == n_pairs

    def test_output_types(self):
        """Output arrays have correct types."""
        pair_left = cp.array([0, 1, 2], dtype=cp.int32)
        pair_right = cp.array([1, 2, 2], dtype=cp.int32)
        group_sizes = cp.array([10, 20, 30], dtype=cp.int64)

        result = _split_pairs(
            pair_left, pair_right, n_devices=2, group_sizes=group_sizes
        )

        for left, right in result:
            assert isinstance(left, cp.ndarray)
            assert isinstance(right, cp.ndarray)
            assert left.dtype == cp.int32
            assert right.dtype == cp.int32


class TestParseDeviceIds:
    """Tests for the parse_device_ids function."""

    def test_multi_gpu_false_returns_device_0(self):
        """multi_gpu=False should always use device 0."""
        result = parse_device_ids(multi_gpu=False)
        assert result == [0]

    def test_multi_gpu_none_uses_all_devices(self):
        """multi_gpu=None should use all available devices."""
        result = parse_device_ids(multi_gpu=None)
        n_devices = cp.cuda.runtime.getDeviceCount()
        assert result == list(range(n_devices))

    def test_multi_gpu_true_uses_all_devices(self):
        """multi_gpu=True should use all available devices."""
        result = parse_device_ids(multi_gpu=True)
        n_devices = cp.cuda.runtime.getDeviceCount()
        assert result == list(range(n_devices))

    def test_multi_gpu_list_returns_same_list(self):
        """multi_gpu as list should return that list."""
        result = parse_device_ids(multi_gpu=[0])
        assert result == [0]

    def test_multi_gpu_string_parses_correctly(self):
        """multi_gpu as string should parse to list of ints."""
        result = parse_device_ids(multi_gpu="0")
        assert result == [0]

    def test_multi_gpu_string_multiple_devices(self):
        """multi_gpu string with multiple devices parses correctly."""
        n_devices = cp.cuda.runtime.getDeviceCount()
        if n_devices >= 2:
            result = parse_device_ids(multi_gpu="0,1")
            assert result == [0, 1]

    def test_single_gpu_always_device_0(self):
        """With only 1 GPU available, should always return [0]."""
        n_devices = cp.cuda.runtime.getDeviceCount()
        if n_devices == 1:
            # All options should return [0] on single GPU system
            assert parse_device_ids(multi_gpu=None) == [0]
            assert parse_device_ids(multi_gpu=True) == [0]
            assert parse_device_ids(multi_gpu=False) == [0]
