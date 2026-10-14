import dataclasses
import unittest

import numpy as np

from site_analysis import TransitionEvents
from site_analysis.transition_events import TransitionEvent


def make_events(timestep=(10, 20, 20), entry_timestep=(10, 10, 20)):
    """Three events: atom 3 moves 7 -> 15 at frame 1, atoms 3 and 9 move at frame 2.

    Atom 3 entered site 7 at frame 1, before committing to it at frame 2.
    """
    return TransitionEvents(
        frame=[1, 2, 2], timestep=timestep, entry_frame=[1, 1, 2],
        entry_timestep=entry_timestep, atom=[3, 3, 9],
        from_site=[7, 15, 2], to_site=[15, 7, 4])


def make_two(**overrides):
    """Two events with no timesteps; keyword arguments replace fields."""
    fields = dict(frame=[1, 2], timestep=None, entry_frame=[1, 2],
                  entry_timestep=None, atom=[3, 3], from_site=[7, 15], to_site=[15, 7])
    fields.update(overrides)
    return TransitionEvents(**fields)


class TransitionEventsTestCase(unittest.TestCase):

    def test_fields_are_read_only_integer_arrays(self):
        events = make_events()
        for name in ("frame", "timestep", "entry_frame", "entry_timestep",
                     "atom", "from_site", "to_site"):
            array = getattr(events, name)
            self.assertIsInstance(array, np.ndarray)
            self.assertEqual(array.dtype, np.int64)
            self.assertFalse(array.flags.writeable)

    def test_input_arrays_are_copied(self):
        frame = np.array([1, 2])
        events = make_two(frame=frame)
        frame[0] = 99
        self.assertEqual(events.frame[0], 1)

    def test_len(self):
        self.assertEqual(len(make_events()), 3)

    def test_integer_index_gives_a_record_of_plain_ints(self):
        event = make_events()[1]
        self.assertEqual(event, TransitionEvent(
            frame=2, timestep=20, entry_frame=1, entry_timestep=10,
            atom=3, from_site=15, to_site=7))
        for name in ("frame", "timestep", "entry_frame", "entry_timestep",
                     "atom", "from_site", "to_site"):
            self.assertIs(type(getattr(event, name)), int)

    def test_record_cannot_be_modified(self):
        event = make_events()[0]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            event.frame = 0

    def test_records_with_equal_values_are_equal(self):
        self.assertEqual(make_events()[0], make_events()[0])
        self.assertNotEqual(make_events()[0], make_events()[1])

    def test_numpy_integer_index_gives_a_record(self):
        events = make_events()
        index = np.flatnonzero(events.atom == 9)[0]
        self.assertEqual(events[index].to_site, 4)

    def test_negative_index(self):
        self.assertEqual(make_events()[-1].atom, 9)

    def test_index_out_of_range_raises(self):
        with self.assertRaises(IndexError):
            make_events()[3]

    def test_timesteps_none_when_absent(self):
        events = make_events(timestep=None, entry_timestep=None)
        self.assertIsNone(events.timestep)
        self.assertIsNone(events.entry_timestep)
        self.assertIsNone(events[0].timestep)
        self.assertIsNone(events[0].entry_timestep)
        self.assertIsNone(events[1:].timestep)
        self.assertIsNone(events[1:].entry_timestep)

    def test_slice_gives_events(self):
        subset = make_events()[1:]
        self.assertIsInstance(subset, TransitionEvents)
        np.testing.assert_array_equal(subset.frame, [2, 2])
        np.testing.assert_array_equal(subset.timestep, [20, 20])
        np.testing.assert_array_equal(subset.entry_frame, [1, 2])
        np.testing.assert_array_equal(subset.entry_timestep, [10, 20])
        np.testing.assert_array_equal(subset.atom, [3, 9])

    def test_boolean_mask_gives_events(self):
        events = make_events()
        into_seven = events[events.to_site == 7]
        self.assertIsInstance(into_seven, TransitionEvents)
        self.assertEqual(len(into_seven), 1)
        self.assertEqual(into_seven[0].from_site, 15)
        self.assertEqual(into_seven[0].entry_frame, 1)

    def test_index_array_gives_events(self):
        subset = make_events()[np.array([2, 0])]
        np.testing.assert_array_equal(subset.atom, [9, 3])
        np.testing.assert_array_equal(subset.entry_frame, [2, 1])

    def test_iteration_gives_records(self):
        records = list(make_events())
        self.assertEqual(len(records), 3)
        self.assertEqual(records[2], TransitionEvent(
            frame=2, timestep=20, entry_frame=2, entry_timestep=20,
            atom=9, from_site=2, to_site=4))

    def test_empty(self):
        events = TransitionEvents(frame=[], timestep=[], entry_frame=[], entry_timestep=[],
                                  atom=[], from_site=[], to_site=[])
        self.assertEqual(len(events), 0)
        self.assertEqual(list(events), [])
        self.assertEqual(events.frame.dtype, np.int64)
        self.assertEqual(events.entry_frame.dtype, np.int64)

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            make_two(atom=[3])

    def test_mismatched_entry_frame_length_raises(self):
        with self.assertRaises(ValueError):
            make_two(entry_frame=[1])

    def test_mismatched_timestep_length_raises(self):
        with self.assertRaises(ValueError):
            make_two(timestep=[10])

    def test_mismatched_entry_timestep_length_raises(self):
        with self.assertRaises(ValueError):
            make_two(entry_timestep=[10])

    def test_two_dimensional_array_raises(self):
        with self.assertRaises(ValueError):
            TransitionEvents(frame=[[1, 2]], timestep=None, entry_frame=[[1, 2]],
                             entry_timestep=None, atom=[[3, 3]],
                             from_site=[[7, 15]], to_site=[[15, 7]])

    def test_positional_arguments_are_rejected(self):
        with self.assertRaises(TypeError):
            TransitionEvents([1], None, [1], None, [3], [7], [15])

    def test_fields_cannot_be_reassigned(self):
        events = make_events()
        with self.assertRaises(AttributeError):
            events.frame = np.array([0, 0, 0])


if __name__ == "__main__":
    unittest.main()
