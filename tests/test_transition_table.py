import unittest

import numpy as np

from site_analysis.transition_table import TransitionTable


class TransitionTableConstructionTestCase(unittest.TestCase):
    """Tests for TransitionTable construction and validation."""

    def test_construction_with_str_keys_and_int_data(self):
        """Test construction with string keys and integer data."""
        data = np.array([[0, 3], [2, 0]])
        table = TransitionTable(keys=["A", "B"], data=data)
        self.assertEqual(table.keys, ["A", "B"])
        np.testing.assert_array_equal(table.data, data)

    def test_construction_with_int_keys_and_float_data(self):
        """Test construction with integer keys and float data."""
        data = np.array([[0.0, 0.75], [1.0, 0.0]])
        table = TransitionTable(keys=[0, 1], data=data)
        self.assertEqual(table.keys, [0, 1])
        np.testing.assert_array_equal(table.data, data)

    def test_non_square_data_raises(self):
        """Test that non-square data raises ValueError."""
        with self.assertRaises(ValueError):
            TransitionTable(keys=["A", "B"], data=np.array([[1, 2, 3], [4, 5, 6]]))

    def test_mismatched_key_length_raises(self):
        """Test that mismatched key length raises ValueError."""
        with self.assertRaises(ValueError):
            TransitionTable(keys=["A", "B", "C"], data=np.array([[0, 1], [2, 0]]))

    def test_duplicate_keys_raises(self):
        """Test that duplicate keys raise ValueError."""
        with self.assertRaises(ValueError):
            TransitionTable(keys=["A", "A"], data=np.array([[0, 1], [2, 0]]))

    def test_frozen(self):
        """Test that TransitionTable is immutable."""
        table = TransitionTable(keys=["A", "B"], data=np.array([[0, 1], [2, 0]]))
        with self.assertRaises(AttributeError):
            table.keys = ["C", "D"]

    def test_empty_table(self):
        """Test construction with empty keys and data."""
        table = TransitionTable(keys=[], data=np.empty((0, 0)))
        self.assertEqual(table.keys, [])
        self.assertEqual(table.data.shape, (0, 0))


class TransitionTableMatrixTestCase(unittest.TestCase):
    """Tests for the .matrix property."""

    def test_matrix_returns_data(self):
        """Test that .matrix returns the underlying ndarray."""
        data = np.array([[0, 3], [2, 0]])
        table = TransitionTable(keys=["A", "B"], data=data)
        np.testing.assert_array_equal(table.matrix, data)


class TransitionTableLocTestCase(unittest.TestCase):
    """Tests for key-based .loc access."""

    def setUp(self):
        self.data = np.array([[0, 3, 1], [2, 0, 4], [5, 6, 0]])
        self.table = TransitionTable(keys=["A", "B", "C"], data=self.data)

    def test_loc_str_keys(self):
        """Test .loc access with string keys."""
        self.assertEqual(self.table.loc["A", "B"], 3)
        self.assertEqual(self.table.loc["B", "A"], 2)
        self.assertEqual(self.table.loc["C", "A"], 5)

    def test_loc_int_keys(self):
        """Test .loc access with integer keys."""
        table = TransitionTable(keys=[10, 20], data=np.array([[0, 7], [3, 0]]))
        self.assertEqual(table.loc[10, 20], 7)
        self.assertEqual(table.loc[20, 10], 3)

    def test_loc_invalid_key_raises_key_error(self):
        """Test that .loc with an invalid key raises KeyError."""
        with self.assertRaises(KeyError):
            self.table.loc["A", "Z"]
        with self.assertRaises(KeyError):
            self.table.loc["Z", "A"]


class TransitionTableToDictTestCase(unittest.TestCase):
    """Tests for .to_dict() conversion."""

    def test_to_dict_str_keys(self):
        """Test .to_dict() with string keys."""
        data = np.array([[0, 3, 1], [2, 0, 0], [0, 4, 0]])
        table = TransitionTable(keys=["A", "B", "C"], data=data)
        self.assertEqual(table.to_dict(), {
            "A": {"A": 0, "B": 3, "C": 1},
            "B": {"A": 2, "B": 0, "C": 0},
            "C": {"A": 0, "B": 4, "C": 0},
        })

    def test_to_dict_int_keys(self):
        """Test .to_dict() with integer keys."""
        data = np.array([[0, 5], [2, 0]])
        table = TransitionTable(keys=[0, 1], data=data)
        self.assertEqual(table.to_dict(), {
            0: {0: 0, 1: 5},
            1: {0: 2, 1: 0},
        })

    def test_to_dict_float_data(self):
        """Test .to_dict() with float data preserves float type."""
        data = np.array([[0.0, 0.75], [1.0, 0.0]])
        table = TransitionTable(keys=["A", "B"], data=data)
        result = table.to_dict()
        self.assertIsInstance(result["A"]["B"], float)
        self.assertAlmostEqual(result["A"]["B"], 0.75)

    def test_to_dict_empty(self):
        """Test .to_dict() with empty table."""
        table = TransitionTable(keys=[], data=np.empty((0, 0)))
        self.assertEqual(table.to_dict(), {})


class TransitionTableReprTestCase(unittest.TestCase):
    """Tests for __repr__."""

    def test_repr_includes_keys_and_shape(self):
        """Test that repr includes keys and matrix shape."""
        table = TransitionTable(keys=["A", "B"], data=np.array([[0, 1], [2, 0]]))
        r = repr(table)
        self.assertIn("A", r)
        self.assertIn("B", r)
        self.assertIn("2", r)


if __name__ == '__main__':
    unittest.main()
