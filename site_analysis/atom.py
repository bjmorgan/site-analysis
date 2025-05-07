from __future__ import annotations

import itertools
import json
from monty.io import zopen # type: ignore
import numpy as np
from pymatgen.core import Structure
from typing import Optional, Any, Union

class Atom(object):
    """Represents a single persistent atom during a simulation.

    Attributes:
        index (int): Unique numeric index identifying this atom.
        in_site (int): Site index for the site this atom
            currently occupies.
        frac_coords (np.array): Numpy array containing the current fractional
            coordinates for this atom.
        trajectory (list): list of site indices occupied at each timestep.

    Note:
        The atom index is used to identify it when parsing structures, so
        needs to be e.g. the corresponding Site index in a Pymatgen Structure.
        
    """

    def __init__(self,
            index: int,
            species_string: Optional[str]=None) -> None:
        """Initialise an Atom object.
        
        Args:
            index (int): Numerical index for this atom. Used to identify this atom
                in analysed structures.
            species_string (Optional[str]): String identifying the chemical species of this atom.
        
        Returns:
            None
        """
        self.index = index
        self.in_site: Optional[int] = None
        self._frac_coords: Optional[np.ndarray] = None
        self.trajectory: list[int] = []
        self.species_string = species_string

    def __str__(self) -> str:
        """Return a string representation of this atom.

        Args:
            None

        Returns:
            (str)

        """
        string = f"Atom: {self.index}"
        return string

    def __repr__(self) -> str:
        string = (
            "site_analysis.Atom("
            f"index={self.index}, "
            f"in_site={self.in_site}, "
            f"frac_coords={self._frac_coords}, "
            f"species_string={self.species_string})"
        )
        return string
        
    def reset(self) -> None:
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
    def frac_coords(self) -> np.ndarray:
        """Getter for the fractional coordinates of this atom.

        Raises:
            AttributeError: if the fractional coordinates for this atom have
                not been set.

        """
        if self._frac_coords is None:
            raise AttributeError("Coordinates not set for atom {}".format(self.index))
        else:
            return self._frac_coords

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "index": int(self.index),
            "in_site": None if self.in_site is None else int(self.in_site),
        }
        if self._frac_coords is not None:
            d["frac_coords"] = self._frac_coords.tolist()
        if hasattr(self, "species_string") and self.species_string is not None:
            d["species_string"] = self.species_string
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> Atom:
        atom = cls(index=d["index"], species_string=d.get("species_string"))
        if d["in_site"] is not None:
            atom.in_site = int(d["in_site"])
        else:
            atom.in_site = None
        atom._frac_coords = np.array(d["frac_coords"])
        return atom

    def to(self,
            filename: Optional[str]=None) -> str:
        s = json.dumps(self.as_dict())
        if filename:
            with zopen(filename, "wb") as f:
                f.write(s.encode('utf-8'))
        return s

    @classmethod
    def from_str(cls,
            input_string: str) -> Atom:
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


def atoms_from_species_string(
        structure:Structure,
        species_string: str) -> list[Atom]:
    atoms = [
        Atom(index=i)
        for i, s in enumerate(structure)
        if s.species_string == species_string
    ]
    return atoms
    
def atoms_from_structure(
    structure: Structure,
    species_string: Union[list[str], str]) -> list[Atom]:
    if isinstance(species_string, str):
        species_string = [species_string]
    atoms = [
        Atom(index=i, species_string=s.species_string)
        for i, s in enumerate(structure)
        if s.species_string in species_string
    ]
    for atom in atoms:
        atom._frac_coords = structure[atom.index].frac_coords
    return atoms

def atoms_from_indices(
        indices: list[int]) -> list[Atom]:
    return [Atom(index=i) for i in indices]
