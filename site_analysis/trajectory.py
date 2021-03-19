from collections import Counter
from tqdm import tqdm, tqdm_notebook # type: ignore
from .polyhedral_site_collection import PolyhedralSiteCollection
from .polyhedral_site import PolyhedralSite
from .voronoi_site import VoronoiSite
from .voronoi_site_collection import VoronoiSiteCollection
from .spherical_site import SphericalSite
from .spherical_site_collection import SphericalSiteCollection

class Trajectory(object):
    """Class for performing sites analysis on simulation trajectories."""

    site_collection_types = { PolyhedralSite: PolyhedralSiteCollection,
                              VoronoiSite: VoronoiSiteCollection,
                              SphericalSite: SphericalSiteCollection }

    def __init__(self, sites, atoms):
        site_collection_class = None
        for k, v in Trajectory.site_collection_types.items():
            if all( [ isinstance( s, k ) for s in sites ] ):
                site_collection_class = v
        if not site_collection_class:
            raise ValueError('Unable to detect unique SiteCollection class')
        self.sites = sites
        self.atoms = atoms
        self.site_collection = site_collection_class(sites)
        self.timesteps = []
        self.previous_occupations = {}
        self.atom_lookup = {a.index: i for i, a in enumerate(atoms)}
        self.site_lookup = {s.index: i for i, s in enumerate(sites)}

    def atom_by_index(self, i):
        return self.atoms[self.atom_lookup[i]] 

    def site_by_index(self, i):
        return self.sites[self.site_lookup[i]] 

    def analyse_structure(self, structure):
        self.site_collection.analyse_structure(self.atoms, structure)
        
    def assign_site_occupations(self, structure):
        self.site_collection.assign_site_occupations(self.atoms, structure)
                    
    def site_coordination_numbers(self):
        return Counter( [ s.coordination_number for s in self.sites ] )

    def site_labels(self):
        return [ s.label for s in self.sites ]
   
    @property
    def atom_sites(self):
        return [ atom.in_site for atom in self.atoms ]
        
    @property
    def site_occupations(self):
        return [ s.contains_atoms for s in self.sites ]

    def append_timestep(self, structure, t=None):
        self.analyse_structure(structure)
        for atom in self.atoms:
            atom.trajectory.append(atom.in_site)
        for site in self.sites:
            site.trajectory.append(site.contains_atoms)
        self.timesteps.append(t)

    def reset(self):
        for atom in self.atoms:
            atom.reset()
        for site in self.sites:
            site.reset()
        self.timesteps = [] 

    @property
    def atoms_trajectory(self):
        return list(map(list, zip(*[atom.trajectory for atom in self.atoms])))

    @property
    def sites_trajectory(self):
        return list(map(list, zip(*[site.trajectory for site in self.sites])))

    @property
    def at(self):
        return self.atoms_trajectory

    @property
    def st(self):
        return self.sites_trajectory

    def trajectory_from_structures(self, structures, progress=False):
        generator = enumerate( structures, 1 )
        if progress:
            if progress=='notebook':
                generator = tqdm_notebook(generator, total=len(structures), unit=' steps')
            else:
                generator = tqdm(generator, total=len(structures), unit=' steps')
        for timestep, s in generator:
            self.append_timestep( s, t=timestep )
   
    def __len__(self):
        """Returns the "length" of a trajectory, i.e. the number of analysed timesteps."""
        return len(self.timesteps)
 
def update_occupation( site, atom ):
    site.contains_atoms.append( atom.index )
    atom.in_site = site.index

