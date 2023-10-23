.. include:: ../substitutions.rst

.. _cli-match_template:

=================
Template matching
=================

TL;DR
^^^^^

The ``match_template.py`` command-line tool facilitates template matching using various scoring metrics. For additional help see :ref:`command-line-options`, :doc:`source code <match_template_code>` or:

.. code-block:: bash

    match_template.py --help

.. note::

    ``match_template.py`` expects high densities to have positive, low densities to have negative values. Therefore, input tomograms typically need to be inverted by multiplication with negative one.

Background
^^^^^^^^^^

Template matching is used under many names in the electron microscopy field:

- Fitting an atomic structure to an electron density map
- Particle picking on tomograms / tilt series / images
- Alignment in subtomogram averaging

Despite their naming, all of these are the same in terms of template matching. The goal is to optimize the similarity between a template and a target. Typically, the template is an atomic structure or an electron density, while the target is almost always an electron density, e.g. a reconstruction or a tomogram.


Configuration
^^^^^^^^^^^^^

The following provides a background on some of the parameters of ``match_template.py``. For simplicity, the arguments were characterized based on whether they relate to the input, performance or accuracy of ``match_template.py``.

Input
-----

- **target/template**: Specifies the paths to your target and template files, which are essential inputs for `match_template`.

- **target_mask/template_mask**: Define masks for the target and template, allowing the tool to focus only on regions of interest and potentially improve accuracy.

Performance
-----------

- **cutoff_target/cutoff_template**: By defining a minimal enclosing box containing all significant density values, the execution time can be significantly improved.

- **no_edge_padding**: By default, each dimension of the target is padded by its mean using the template shape. Avoiding this padding can speed up computations, but users should be cautious as it might lead to boundary inaccuracies.

- **no_fourier_padding**: While zero-padding to the full convolution shape ensures numerical stability, avoiding it can lead to faster computations. Users might occasionally observe inaccuracies with this option turned off.

- **interpolation_order**: Defines the spline interpolation's granularity during rotations. While higher values ensure more accuracy, especially on CPU, reducing it can lead to performance improvements at a slight accuracy trade-off.

- **use_mixed_precision**: By utilizing float16 for GPU operations, memory consumption is lowered, and certain hardware might observe a performance boost.

- **use_memmap**: When working with large inputs, this option lets the tool offload objects to the disk, making computations feasible even with limited RAM, though this comes with a slight IO performance overhead.


Accuracy
--------

- **score**: The choice of the scoring function plays a pivotal role in the accuracy of results. Users can select from a variety of options, as detailed in the section :doc:`../reference/matching_exhaustive`.

- **angular_sampling**: Determines the granularity of the angular sampling for fitting. A finer sampling (lower value) will be more exhaustive, potentially yielding more accurate results but at a computational cost.

- **scramble_phases**: This option scrambles the phases of the templates to simulate a noise background, aiding in refining the score space.


Usage Examples
^^^^^^^^^^^^^^

The following outlines some typical use cases of ``match_template.py``.

.. tabs::

    .. tab:: Particle Picking

        When performing particle picking, the goal is to identify and isolate specific particles (in this case, ribosomes) from a larger dataset (tomogram). By using a known structure as a template, ``match_template.py`` can scan the tomogram to find locations where this structure is likely to appear.

        **Data sources:**

        - Target:
            1. Navigate to the `EMPIAR database <https://www.ebi.ac.uk/empiar/EMPIAR-10988/>`_ and look for the accession EMPIAR-10988.
            2. Locate the section titled "2. Reconstructed cryo-electron tomograms acquired with defocus-only (DEF) on S. pombe cryo-FIB lamellae".
            3. Manually download the file named `TS_037.rec`. It will be in a ZIP archive.
            4. Once downloaded, unzip the file to access the tomogram.

        - Template:
            The template used for ribosome structures is `EMD-3228 <https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-3228/map/emd_3228.map.gz>`_.

            .. code-block:: bash

                wget https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-3228/map/emd_3228.map.gz

        **Procedure**:

        Invert the input tomogram so that high densities have positive, low densities negative values.

        .. code-block:: python

            from tme import Density
            density = Density.from_file("TS_037.rec")
            density.data = density.data * -1
            density.to_file("TS_037_inverted.mrc")


        Run ``match_template.py`` to pick ribosomes:

        .. code-block:: bash

          match_template.py \
              -m TS_037_inverted.mrc \
              -i emd_3228.map.gz \
              -n 2 \
              -a 60


    .. tab:: Fit Atomic Structure

        When fitting an atomic structure, the objective is to position a structure or a structure subunit within another density. This method determines the optimal orientation and positioning of the atomic structure within the given density.

        **Data sources:**

        - Target:
            * **Web Link:** `EMD-8621 Map <https://www.ebi.ac.uk/emdb/EMD-8621>`_
            * **Direct Download**:

            .. code-block:: bash

                wget https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-8621/map/emd_8621.map.gz

        - Template:
            * **Web Link:** `5uz4 Structure <https://files.rcsb.org/download/5UZ4.pdb>`_
            * **Direct Download:**

            .. code-block:: bash

                wget https://files.rcsb.org/download/5UZ4.pdb


        **Procedure**:

        To fit an atomic structure into an electron density map:

        .. code-block:: bash

            match_template.py \
                -m emd_8621.map.gz \
                -i 5UZ4.pdb \
                --cutoff_target 0.0309 \
                --cutoff_template 0 \
                -n 4 \
                -a 40 \
                --no_edge_padding


    .. tab:: Alignment of Densities

        For density alignment, the aim is to find the best fit between two densities. This involves determining the optimal orientation and alignment that allows one density to closely match and overlap with another.

        **Procedure**:

        First lets simulate a translation and rotation of the ribosome map:

        .. code-block:: python

            import numpy as np
            from tme import Density

            density = Density.from_file("emd_8621.map.gz")
            new_density, _ = density.centered()
            rotation_matrix = np.eye(3)
            rotation_matrix[0,0] = -1
            rotation_matrix[2,2] = -1

            new_density = new_density.rigid_transform(
              rotation_matrix = rotation_matrix,
              translation = [-2,5,0],
            )
            new_density.adjust_box(
              new_density.trim_box(0.0309)
            )
            new_density.to_file("emd_8621_transformed.mrc")

        To realign the two densities we can use ``match_template.py``:

        .. code-block:: bash

            match_template.py   \
                -m emd_8621.map.gz  \
                -i emd_8621_transformed.mrc \
                --cutoff_target 0.0309 \
                --cutoff_template 0.0309 \
                -a 60 \
                -s CORR \
                -n 4 \
                --no_edge_padding


Conclusion
^^^^^^^^^^

Following this tutorial, you have gained an understanding of template matching using |project| within the realm of electron microscopy. Specifically:

- **Applications**: We explored how template matching functions in diverse scenarios like particle picking, atomic structure fitting, and density alignments.

- **Tool Configuration**: By diving into the `match_template` configurations, we highlighted the balance between accuracy, performance, and input requirements.

- **Practical Usage**: Through specific examples, we showcased how to apply the tool effectively.

In the subsequent tutorial, you will learn how to interpret the output of match_template to extract actionable insigts and optimize the extraction of meaningful information from your electron microscopy data.

.. _command-line-options:

Command-Line Options
^^^^^^^^^^^^^^^^^^^^

.. program:: match_template.py

.. option:: -m <target>, --target <target>

    Path to a target in CCP4/MRC format. This argument is required.

.. option:: --target_mask <target_mask>

    Path to a mask for the target in CCP4/MRC format.

.. option:: --cutoff_target <cutoff_target>

    Target contour level used for cropping.

.. option:: --cutoff_template <cutoff_template>

    Template contour level used for cropping.

.. option:: -i <template>, --template <template>

    Path to a template in PDB/MMCIF or CCP4/MRC format. This argument is required.

.. option:: --template_mask <template_mask>

    Path to a mask for the template in CCP4/MRC format.

.. option:: -o <output>

    Path to output pickle file. Default is "output.pickle".

.. option:: -s <score>

    Template matching scoring function. Choices based on available scoring functions in the code.

.. option:: -n <cores>

    Number of cores used for template matching. Default is 4.

.. option:: -r <ram>, --ram <ram>

    Amount of RAM that can be used in bytes.

.. option:: -a <angular_sampling>

    Angular sampling for fitting. Lower numbers yield more rotations. Default is 40.0.

.. option:: -p

    When set, perform peak calling instead of score aggregation.

.. option:: --use_gpu

    Whether to perform computations on the GPU.

.. option:: --use_gpu

    Comma separated list of GPU devices to use, e.g. 0,1. If not provided and --use_gpu is set, the environment variable CUDA_VISIBLE_DEVICES will be respected.

.. option:: --no_edge_padding

    Whether to pad the edges of the target. Useful if the target has a well-defined bounding box, like a density map.

.. option:: --no_fourier_padding

    Determine if input arrays should be zero-padded to the full convolution shape for numerical stability.

.. option:: --scramble_phases

    Whether to phase scramble the template for subsequent normalization.

.. option:: --interpolation_order <interpolation_order>

    Spline interpolation used during rotations. If less than zero, no interpolation is performed. Default is 3.

.. option:: --use_mixed_precision

    Use float16 for real values operations where possible.

.. option:: --use_memmap

    Use memmaps to offload large data objects to disk. Useful for large inputs without setting the -p flag.

.. option:: --temp_directory <temp_directory>

    Directory for temporary objects. Faster I/O often improves runtime.

.. option:: --gaussian_sigma

    Sigma parameter for Gaussian filtering the template.

.. option:: --bandpass_band

    Comma separated start and stop frequency for bandpass filtering the template, e.g. 0.1, 0.5.

.. option:: --bandpass_smooth

    Smooth parameter for the bandpass filter.

.. option:: --tilt_range

    Comma separated start and stop stage tilt angle, e.g. 50,45. Used to create a wedge mask to be applied to the template.

.. option:: --tilt_step

    Step size between tilts, e.g. 5. When set a more accurate wedge mask will be computed.

.. option:: --wedge_smooth

    Gaussian sigma used to smooth the wedge mask.
