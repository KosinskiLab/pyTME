Fit a protein structure to an electron density
==============================================

In this guide, we will walk you through the process of using template matching to fit a protein structure to an electron density map. This involves acquiring the necessary map and structure data and then applying our API's functions to achieve the match.

Obtain the Necessary Files
--------------------------

Head over to the `European Bioinformatics Institute's (EMBL-EBI) repository <https://www.ebi.ac.uk/emdb/EMD-8621>`_ to visually inspect and understand the data we will be working with.

For the purpose of this guide, we will need the following files:

1. **Map File** - This contains the electron density of our molecule.

   * **Web Link:** `EMD-8621 Map <https://www.ebi.ac.uk/emdb/EMD-8621>`_
   * **Direct Download:** If you're familiar with command-line utilities like `wget` or `curl`, you can directly download using the link below:

     .. code-block:: bash

        wget https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-8621/map/emd_8621.map.gz

2. **Structure File** - This is the protein structure that we want to match to our map.

   * **Web Link:** `5uz4 Structure <https://files.rcsb.org/download/5UZ4.pdb>`_
   * **Direct Download:** Again, if you're using command-line, use the following:

     .. code-block:: bash

        wget https://files.rcsb.org/download/5UZ4.pdb

Perform template matching
-------------------------

We are going to use the functionalities integrated in the :py:class:`tme.density.Density` class to perform non-exhaustive template matching using :py:class:`tme.matching_optimization.NormalizedCrossCorrelation`.

In the following code, we make use of the data structures :py:class:`tme.density.Density` and :py:class:`tme.structure.Structure`, which hold the electron density maps and the atomic structure respectively. Both are passed to :py:meth:`tme.density.Density.match_structure_to_density`, which numerically identifies the translation and rotation of the input structure that maximizes the normalized cross correlation score. The aligned structure is written to disk as pdb file using :py:meth:`tme.structure.Structure.to_file`.

.. code-block:: python

    from tme import Structure, Density

    # Paths to downloaded files, adapt to your local paths
    map_file = "emd_8621.map.gz"
    structure_file = "5UZ4.pdb"
    structure_aligned_path = "5UZ4_aligned.pdb"

    # Load the data
    density = Density.from_file(map_file)
    structure = Structure.from_file(
        structure_file,
        filter_by_residues = None # Keep non amino acid residues
    )

    ret = density.match_structure_to_density(
      target = density,
      template = structure,
      cutoff_target = 0.0309, # Author reported contour level
      scoring_method = "NormalizedCrossCorrelation"
    )
    structure_aligned, translation, rotation_matrix = ret

    structure_aligned.to_file(structure_aligned_path)

:py:meth:`tme.density.Density.match_structure_to_density` accepts a :py:class:`tme.structure.Structure` instance which contains atomic coordinates. However, the more common scenario is that one has two data arrays which should be template matched. :py:class:`tme.density.Density` can be conveniently instantiated from an atomic structure file using :py:meth:`tme.density.Density.from_structure`. In this case, it is paramount to pass the sampling_rate parameter to register the atomic structure on a grid with equal spacing than the electron density map.

Depending on the sampling rate the scoring function can be more or less smooth. Desirable is a smooth funnel-like score landscape that allows the underlying optimizer to find the optimal translation and rotation of the input structure. We can smooth the scoring function by applying filters to the atomic stucture template. For this purpose, we used methods from the :py:class:`tme.preprocessor.Preprocessor`, which can directly operate on the data attribute from the :py:class:`tme.density.Density` template instance.

In the following we use :py:meth:`tme.preprocessor.Preprocessor.gaussian_filter` to perform gaussian smoothing of the structure. The output is a numpy array, which we assign to the data attribute of an empty copy of the structure. The smoothed structure is subsequently used as template in :py:meth:`tme.density.Density.match_densities`, which returns the aligned template as well as the used translation and rotation matrix.

.. code-block:: python

    import numpy as np
    from tme import Density, Structure, Preprocessor

    # Paths to downloaded files
    map_file = "emd_8621.map.gz"
    structure_file = "5UZ4.pdb"
    structure_aligned_path = "5UZ4_aligned.mrc"

    density = Density.from_file(map_file)
    # Resampling the map to reduce computation time
    density = density.resample(4 * density.sampling_rate)
    # Load the data into equally spaced grids
    structure_density = Density.from_structure(
        structure_file,
        sampling_rate = density.sampling_rate,
        filter_by_residues = None # Keep non amino acid residues
    )

    preprocessor = Preprocessor()
    structure_smoothed = structure_density.empty
    structure_smoothed.data = preprocessor.gaussian_filter(
      template = structure_density.data,
      sigma = 2
    )

    structure_aligned, translation, rotation_matrix = density.match_densities(
      target = density,
      template = structure_smoothed,
      cutoff_target = 0.0309, # Author reported contour level
      cutoff_template = 0.2,
      scoring_method = "NormalizedCrossCorrelation"
    )

    structure_aligned.to_file(structure_aligned_path)

The number of non-zero elements in structure_smoothed increases with the sigma value, hence the computation time also increases. Consider either fitting a subset of all elements or increasing cutoff_template.

Note that in the code above, structure_aligned contains the smoothed density. To obtain the aligned unsmoothed density you would need to use the determined translation and rotation matrix like so:

.. code-block:: python

   structure_density_out = structure_density.rigid_transform(
      rotation_matrix = rotation_matrix,
      use_geometric_center = False
   )
   structure_density_out.origin = structure_aligned.origin.copy()


Similarly, the template matched structure can be obtained like:

.. code-block:: python

    structure = Structure.from_file(
        structure_file,
        filter_by_residues = None # Keep non amino acid residues
    )
    final_translation = np.add(
        -structure_density.origin,
        np.multiply(translation, structure_density.sampling_rate),
    )

    # Atom coordinates are in xyz
    final_translation = final_translation[::-1]
    rotation_matrix = rotation_matrix[::-1, ::-1]

    structure_out = structure.rigid_transform(
        translation=final_translation, rotation_matrix=rotation_matrix
    )

.. note::

    Structure coordinates are in xyz format, while Densities are zyx. :py:meth:`tme.density.Density.match_densities` operates on densities and hence the output translation and rotation matrix have to be adapted accordingly. The output of :py:meth:`tme.density.Density.match_structure_to_density` however can be directly passed to :py:meth:`tme.structure.Structure.rigid_transform`.

Conclusion
----------

By following the steps above, you have successfully performed template matching of a protein structure against a molecular map. Feel free to explore more functionalities provided by our API!
