.. include:: ../../substitutions.rst
.. currentmodule:: tme.matching_utils

Utilities
=========

Functions that are used in exhaustive and non-exhaustive template matching or are specific to template matching in general are grouped in the utilities module.


Masks
~~~~~
.. autosummary::
   :toctree: ../api/

   box_mask
   create_mask
   tube_mask
   elliptical_mask

Subsetting
~~~~~~~~~~
.. autosummary::
   :toctree: ../api/

   centered
   split_shape
   centered_mask
   apply_convolution_mode
   compute_full_convolution_index
   compute_parallelization_schedule


Serialization
~~~~~~~~~~~~~
.. autosummary::
   :toctree: ../api/

   load_pickle
   write_pickle


Utilities
~~~~~~~~~
.. autosummary::
   :toctree: ../api/

   scramble_phases
   rigid_transform
   array_to_memmap
   memmap_to_array
   normalize_template
   conditional_execute
   minimum_enclosing_box
   generate_tempfile_name


.. currentmodule:: tme.rotations

Rotations
~~~~~~~~~
.. autosummary::
   :toctree: ../api/

   align_to_axis
   align_vectors
   euler_to_rotationmatrix
   euler_from_rotationmatrix
   get_cone_rotations
   get_rotation_matrices
