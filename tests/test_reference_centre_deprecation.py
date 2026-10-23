"""Unit tests for the warnings for sites without reference centres."""

import unittest

from site_analysis._reference_centre_deprecation import (
	on_users_behalf, warn_without_reference_centres)
from tests.test_helpers import recorded_future_warnings


def warn():
	"""Issue the warning, as a public function of the package would."""
	warn_without_reference_centres("deprecated")


class TestOnUsersBehalf(unittest.TestCase):
	"""Tests for silencing the warning with on_users_behalf."""

	def test_warns_outside_the_block(self):
		"""The warning is issued outside the block."""
		with recorded_future_warnings() as caught:
			warn()
		self.assertEqual(len(caught), 1)

	def test_silent_inside_the_block(self):
		"""The warning is not issued inside the block."""
		with recorded_future_warnings() as caught:
			with on_users_behalf():
				warn()
		self.assertEqual(caught, [])

	def test_warns_after_an_exception_in_the_block(self):
		"""The warning is issued again after the block exits with an exception."""
		with recorded_future_warnings() as caught:
			with self.assertRaises(RuntimeError):
				with on_users_behalf():
					raise RuntimeError
			warn()
		self.assertEqual(len(caught), 1)

	def test_outer_block_stays_silent_after_a_nested_block_exits(self):
		"""Inside an outer block, the warning stays silent after a nested block exits."""
		with recorded_future_warnings() as caught:
			with on_users_behalf():
				with on_users_behalf():
					pass
				warn()
		self.assertEqual(caught, [])


if __name__ == '__main__':
	unittest.main()
