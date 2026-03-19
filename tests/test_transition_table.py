import unittest

import numpy as np

from site_analysis.transition_table import TransitionTable


class TransitionTableConstructionTestCase(unittest.TestCase):
    """Tests for TransitionTable construction and validation."""

    def test_construction_with_str_keys_and_int_data(self):
        """Test construction with string keys and integer data."""
        matrix = np.array([[0, 3], [2, 0]])
        table = TransitionTable(keys=("A", "B"), matrix=matrix)
        self.assertEqual(table.keys, ("A", "B"))
        np.testing.assert_array_equal(table.matrix, matrix)

    def test_construction_with_int_keys_and_float_data(self):
        """Test construction with integer keys and float data."""
        matrix = np.array([[0.0, 0.75], [1.0, 0.0]])
        table = TransitionTable(keys=(0, 1), matrix=matrix)
        self.assertEqual(table.keys, (0, 1))
        np.testing.assert_array_equal(table.matrix, matrix)

    def test_non_square_data_raises(self):
        """Test that non-square data raises ValueError."""
        with self.assertRaises(ValueError):
            TransitionTable(keys=("A", "B"), matrix=np.array([[1, 2, 3], [4, 5, 6]]))

    def test_mismatched_key_length_raises(self):
        """Test that mismatched key length raises ValueError."""
        with self.assertRaises(ValueError):
            TransitionTable(keys=("A", "B", "C"), matrix=np.array([[0, 1], [2, 0]]))

    def test_duplicate_keys_raises(self):
        """Test that duplicate keys raise ValueError."""
        with self.assertRaises(ValueError):
            TransitionTable(keys=("A", "A"), matrix=np.array([[0, 1], [2, 0]]))

    def test_frozen(self):
        """Test that TransitionTable is immutable."""
        table = TransitionTable(keys=("A", "B"), matrix=np.array([[0, 1], [2, 0]]))
        with self.assertRaises(AttributeError):
            table.keys = ("C", "D")

    def test_matrix_is_read_only(self):
        """Test that the matrix array is not writeable."""
        table = TransitionTable(keys=("A", "B"), matrix=np.array([[0, 1], [2, 0]]))
        with self.assertRaises(ValueError):
            table.matrix[0, 0] = 99

    def test_empty_table(self):
        """Test construction with empty keys and data."""
        table = TransitionTable(keys=(), matrix=np.empty((0, 0)))
        self.assertEqual(table.keys, ())
        self.assertEqual(table.matrix.shape, (0, 0))


class TransitionTableGetTestCase(unittest.TestCase):
    """Tests for key-based .get() access."""

    def setUp(self):
        self.matrix = np.array([[0, 3, 1], [2, 0, 4], [5, 6, 0]])
        self.table = TransitionTable(keys=("A", "B", "C"), matrix=self.matrix)

    def test_get_str_keys(self):
        """Test .get() access with string keys."""
        self.assertEqual(self.table.get("A", "B"), 3)
        self.assertEqual(self.table.get("B", "A"), 2)
        self.assertEqual(self.table.get("C", "A"), 5)

    def test_get_int_keys(self):
        """Test .get() access with integer keys."""
        table = TransitionTable(keys=(10, 20), matrix=np.array([[0, 7], [3, 0]]))
        self.assertEqual(table.get(10, 20), 7)
        self.assertEqual(table.get(20, 10), 3)

    def test_get_invalid_key_raises_key_error(self):
        """Test that .get() with an invalid key raises KeyError."""
        with self.assertRaises(KeyError):
            self.table.get("A", "Z")
        with self.assertRaises(KeyError):
            self.table.get("Z", "A")


class TransitionTableToDictTestCase(unittest.TestCase):
    """Tests for .to_dict() conversion."""

    def test_to_dict_str_keys(self):
        """Test .to_dict() with string keys."""
        matrix = np.array([[0, 3, 1], [2, 0, 0], [0, 4, 0]])
        table = TransitionTable(keys=("A", "B", "C"), matrix=matrix)
        self.assertEqual(table.to_dict(), {
            "A": {"A": 0, "B": 3, "C": 1},
            "B": {"A": 2, "B": 0, "C": 0},
            "C": {"A": 0, "B": 4, "C": 0},
        })

    def test_to_dict_int_keys(self):
        """Test .to_dict() with integer keys."""
        matrix = np.array([[0, 5], [2, 0]])
        table = TransitionTable(keys=(0, 1), matrix=matrix)
        self.assertEqual(table.to_dict(), {
            0: {0: 0, 1: 5},
            1: {0: 2, 1: 0},
        })

    def test_to_dict_float_data(self):
        """Test .to_dict() with float data preserves float type."""
        matrix = np.array([[0.0, 0.75], [1.0, 0.0]])
        table = TransitionTable(keys=("A", "B"), matrix=matrix)
        result = table.to_dict()
        self.assertIsInstance(result["A"]["B"], float)
        self.assertAlmostEqual(result["A"]["B"], 0.75)

    def test_to_dict_empty(self):
        """Test .to_dict() with empty table."""
        table = TransitionTable(keys=(), matrix=np.empty((0, 0)))
        self.assertEqual(table.to_dict(), {})


class TransitionTableReorderTestCase(unittest.TestCase):
    """Tests for .reorder() method."""

    def test_reorder_permutes_rows_and_columns(self):
        """Test that reorder permutes rows and columns correctly."""
        matrix = np.array([[0, 3, 1], [2, 0, 4], [5, 6, 0]])
        table = TransitionTable(keys=("A", "B", "C"), matrix=matrix)
        reordered = table.reorder(["C", "A", "B"])
        self.assertEqual(reordered.keys, ("C", "A", "B"))
        np.testing.assert_array_equal(reordered.matrix, np.array([
            [0, 5, 6],
            [1, 0, 3],
            [4, 2, 0],
        ]))

    def test_reorder_missing_key_raises(self):
        """Test that omitting a key raises ValueError."""
        table = TransitionTable(keys=("A", "B", "C"), matrix=np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            table.reorder(["A", "B"])

    def test_reorder_extra_key_raises(self):
        """Test that adding an unknown key raises ValueError."""
        table = TransitionTable(keys=("A", "B"), matrix=np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            table.reorder(["A", "B", "C"])


class TransitionTableFilterTestCase(unittest.TestCase):
    """Tests for .filter() method."""

    def setUp(self):
        self.matrix = np.array([
            [0, 3, 1],
            [2, 0, 4],
            [5, 6, 0],
        ])
        self.table = TransitionTable(keys=("A", "B", "C"), matrix=self.matrix)

    def test_basic_subset(self):
        """Test filtering to a subset of keys."""
        filtered = self.table.filter(["A", "C"])
        self.assertEqual(filtered.keys, ("A", "C"))
        np.testing.assert_array_equal(filtered.matrix, np.array([
            [0, 1],
            [5, 0],
        ]))

    def test_preserves_requested_order(self):
        """Test that filter respects the order of the provided keys."""
        filtered = self.table.filter(["C", "A"])
        self.assertEqual(filtered.keys, ("C", "A"))
        np.testing.assert_array_equal(filtered.matrix, np.array([
            [0, 5],
            [1, 0],
        ]))

    def test_single_key(self):
        """Test filtering to a single key gives a 1x1 table."""
        filtered = self.table.filter(["B"])
        self.assertEqual(filtered.keys, ("B",))
        np.testing.assert_array_equal(filtered.matrix, np.array([[0]]))

    def test_all_keys(self):
        """Test filtering with all keys returns an equivalent table."""
        filtered = self.table.filter(["A", "B", "C"])
        self.assertEqual(filtered, self.table)

    def test_empty_keys(self):
        """Test filtering with empty keys returns an empty table."""
        filtered = self.table.filter([])
        self.assertEqual(filtered.keys, ())
        self.assertEqual(filtered.matrix.shape, (0, 0))

    def test_unknown_key_raises(self):
        """Test that filtering with a key not in the table raises ValueError."""
        with self.assertRaises(ValueError):
            self.table.filter(["A", "Z"])

    def test_duplicate_key_raises(self):
        """Test that filtering with duplicate keys raises ValueError."""
        with self.assertRaises(ValueError):
            self.table.filter(["A", "A"])

    def test_filter_then_reorder(self):
        """Test chaining filter() then reorder()."""
        result = self.table.filter(["A", "C"]).reorder(["C", "A"])
        self.assertEqual(result.keys, ("C", "A"))
        np.testing.assert_array_equal(result.matrix, np.array([
            [0, 5],
            [1, 0],
        ]))


class TransitionTableEqualityTestCase(unittest.TestCase):
    """Tests for __eq__."""

    def test_equal_tables(self):
        """Test that identical tables are equal."""
        m = np.array([[0, 1], [2, 0]])
        t1 = TransitionTable(keys=("A", "B"), matrix=m.copy())
        t2 = TransitionTable(keys=("A", "B"), matrix=m.copy())
        self.assertEqual(t1, t2)

    def test_different_keys_not_equal(self):
        """Test that tables with different keys are not equal."""
        m = np.array([[0, 1], [2, 0]])
        t1 = TransitionTable(keys=("A", "B"), matrix=m.copy())
        t2 = TransitionTable(keys=("X", "Y"), matrix=m.copy())
        self.assertNotEqual(t1, t2)

    def test_different_data_not_equal(self):
        """Test that tables with different data are not equal."""
        t1 = TransitionTable(keys=("A", "B"), matrix=np.array([[0, 1], [2, 0]]))
        t2 = TransitionTable(keys=("A", "B"), matrix=np.array([[0, 9], [2, 0]]))
        self.assertNotEqual(t1, t2)

    def test_not_equal_to_non_table(self):
        """Test that comparison with non-TransitionTable returns NotImplemented."""
        table = TransitionTable(keys=("A", "B"), matrix=np.array([[0, 1], [2, 0]]))
        self.assertNotEqual(table, "not a table")

    def test_not_hashable(self):
        """Test that TransitionTable is not hashable."""
        table = TransitionTable(keys=("A", "B"), matrix=np.array([[0, 1], [2, 0]]))
        with self.assertRaises(TypeError):
            hash(table)


class TransitionTableReprTestCase(unittest.TestCase):
    """Tests for __repr__."""

    def test_repr_includes_keys_and_shape(self):
        """Test that repr includes keys and matrix shape."""
        table = TransitionTable(keys=("A", "B"), matrix=np.array([[0, 1], [2, 0]]))
        r = repr(table)
        self.assertIn("A", r)
        self.assertIn("B", r)
        self.assertIn("2", r)


if __name__ == '__main__':
    unittest.main()
