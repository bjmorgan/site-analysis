class SiteCollection(object):
    """Parent class for collections of sites.

    Collections of specific site types should inherit from this class.

    Attributes:
        sites (list): List of ``Site``-like objects.

    """

    def __init__(self, sites):
        """Create a SiteCollection object.

        Args:
            sites (list): List of ``Site`` objects.

        """
        self.sites = sites

    def assign_site_occupations(self, atoms, structure):
        """Assigns atoms to sites for a specific structure.

        This method should be implemented in the inherited subclass

        Args:
            atoms (list(Atom)): List of Atom objects to be assigned to sites.
            struture (pymatgen.Structure): Pymatgen Structure object used to specificy
                the atomic coordinates.

        Returns:
            None

        Notes:
            The atom coordinates should already be consistent with the coordinates
            in `structure`. Recommended usage is via the ``analyse_structure()`` method.
        """
        raise NotImplementedError('assign_site_occupations should be implemented in the inherited class')

    def analyse_structure(self, atoms, structure):
        """Perform a site analysis for a set of atoms on a specific structure.

        This method should be implemented in the inherited subclass.

        """
        raise NotImplementedError('analyse_structure should be implemented in the inherited class')

    def neighbouring_sites(self, site_index):
        raise NotImplementedError('neighbouring_sites should be implemented in the inherited class')

    def update_occupation(self, site, atom):
        site.contains_atoms.append( atom.index )
        site.points.append( atom.frac_coords )
        atom.in_site = site.index

    def reset_site_occupations(self):
        for s in self.sites:
            s.contains_atoms = []

    def sites_contain_points(self, structure, points):
        check = all([s.contains_point(p,structure=structure) for s, p in zip(self.sites, points)])
        self.reset_site_occupations()
        return check
  
        

