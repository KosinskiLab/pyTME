Density
=======

.. currentmodule:: tme.density

Constructor
~~~~~~~~~~~

.. autosummary::
   :toctree: api/
   :nosignatures:

   Density

Attributes and underlying data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Density.data
   Density.origin
   Density.sampling_rate
   Density.metadata
   Density.shape
   Density.empty

Serialization
~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Density.to_file
   Density.from_file
   Density.from_structure

Conversion
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Density.copy
   Density.to_numpy
   Density.to_memmap

Subsetting
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Density.pad
   Density.trim_box
   Density.centered
   Density.adjust_box
   Density.minimum_enclosing_box

Computations
~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Density.rigid_transform
   Density.to_pointcloud
   Density.resample
   Density.density_boundary
   Density.surface_coordinates
   Density.normal_vectors
   Density.core_mask
   Density.center_of_mass
   Density.match_densities
   Density.match_structure_to_density
   Density.fourier_shell_correlation
