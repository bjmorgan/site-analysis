import itertools
import json
from monty.io import zopen # type: ignore
import numpy as np
from pymatgen.core import Structure

class Atom(object):
    """Represents a single persistent atom during a simulation.

    Attributes:
        index (int): Unique numeric index identifying this atom.
        in_site (int): Site index for the site this atom
            currently occupies.
        frac_coords (np.array): Numpy array containing the current fractional
            coordinates for this atom.
        trajectory (list): List of site indices occupied at each timestep.

    Note:
        The atom index is used to identify it when parsing structures, so
        needs to be e.g. the corresponding Site index in a Pymatgen Structure.
        
    """

    def __init__(self, index, species_string=None):
        """Initialise an Atom object.

        Args:
            index (int): Numerical index for this atom. Used to identify this atom
                in analysed structures.

        Returns:
            None

        """
        self.index = index
        self.in_site = None
        self._frac_coords = None
        self.trajectory = []

    def __str__(self):
        """Return a string representation of this atom.

        Args:
            None

        Returns:
            (str)

        """
        string = f"Atom: {self.index}"
        return string

    def __repr__(self):
        string = (
            "site_analysis.Atom("
            f"index={self.index}, "
            f"in_site={self.in_site}, "
            f"frac_coords={self._frac_coords})"
        )
        return string

    def reset(self):
        """Reset the state of this Atom.

        Clears the `in_site` and `trajectory` attributes.

        Returns:
            None

        """
        self.in_site = None
        self._frac_coords = None
        self.trajectory = []

    def assign_coords(self,
            structure: Structure) -> None:
        """Assign fractional coordinates to this atom from a 
        pymatgen Structure.

        Args:
            structure (pymatgen.Structure): The Structure to use for this atom's
                fractional coordinates.

        Returns:
            None

        """
        self._frac_coords = structure[self.index].frac_coords

    @property
    def frac_coords(self):
        """Getter for the fractional coordinates of this atom.

        Raises:
            AttributeError: if the fractional coordinates for this atom have
                not been set.

        """
        if self._frac_coords is None:
            raise AttributeError("Coordinates not set for atom {}".format(self.index))
        else:
            return self._frac_coords

    def as_dict(self):
        d = {
            "index": self.index,
            "in_site": self.in_site,
            "frac_coords": self._frac_coords.tolist(),
        }
        return d

    @classmethod
    def from_dict(cls, d):
        atom = cls(index=d["index"])
        atom.in_site = d["in_site"]
        atom._frac_coords = np.array(d["frac_coords"])
        return atom

    def to(self, filename=None):
        s = json.dumps(self.as_dict())
        if filename:
            with zopen(filename, "wt") as f:
                f.write("{}".format(s))
        return s

    @classmethod
    def from_str(cls, input_string):
        """Initiate an Atom object from a JSON-formatted string.

        Args:
            input_string (str): JSON-formatted string.

        Returns:
            (Atom)

        """
        d = json.loads(input_string)
        return cls.from_dict(d)

    @classmethod
    def from_file(cls, filename):
        with zopen(filename, "rt") as f:
            contents = f.read()
        return cls.from_str(contents)


def atoms_from_species_string(structure, species_string):
    atoms = [
        Atom(index=i)
        for i, s in enumerate(structure)
        if s.species_string == species_string
    ]
    return atoms


def atoms_from_indices(indices):
    return [Atom(index=i) for i in indices]
