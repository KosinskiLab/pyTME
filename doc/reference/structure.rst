Structure
=========

.. currentmodule:: tme.structure

Constructor
~~~~~~~~~~~

.. autosummary::
   :toctree: api/
   :nosignatures:

   Structure

Attributes and underlying data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Structure.record_type
   Structure.atom_serial_number
   Structure.atom_name
   Structure.atom_coordinate
   Structure.alternate_location_indicator
   Structure.residue_name
   Structure.chain_identifier
   Structure.residue_sequence_number
   Structure.code_for_residue_insertion
   Structure.occupancy
   Structure.temperature_factor
   Structure.segment_identifier
   Structure.element_symbol
   Structure.charge
   Structure.metadata

Serialization
~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Structure.to_file
   Structure.from_file

Conversion
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Structure.to_volume

Subsetting
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Structure.copy
   Structure.subset_by_chain
   Structure.subset_by_range

Computations
~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Structure.rigid_transform
   Structure.align_structures
   Structure.compare_structures
   Structure.center_of_mass
   Structure.centered
