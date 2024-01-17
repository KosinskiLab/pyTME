.. include:: ../substitutions.rst

.. _cli-match_template:

*****************
Template matching
*****************

TL;DR
=====

The ``match_template.py`` command-line tool facilitates template matching using various scoring metrics. For additional help see :ref:`command-line-options`, :doc:`source code <code_examples/match_template_code>` or:

.. code-block:: bash

    match_template.py --help

.. note::

    ``match_template.py`` expects high densities to have positive (white contrast), low densities to have negative values (black contrast). Therefore, input tomograms typically need to be inverted by multiplication with negative one. Version 0.1.4 introduced the ``--invert_target_contrast`` flag, which automatically performs the inversion for you.

.. _match-template-background:

Background
==========

Template matching is used under many names in the electron microscopy field:

- Fitting an atomic structure to an electron density map

- Particle picking on tomograms / tilt series / images

- Alignment in subtomogram averaging

Despite their naming, all of these follow the same template matching formalism.

The goal of template matching is to optimize the similarity between a template and a target. Typically, the template is an atomic structure or an electron density, while the target is almost always an electron density, e.g. a reconstruction or a tomogram. By default ``match_template.py`` measures similarity using the scoring method :py:meth:`tme.matching_exhaustive.flcSphericalMask_setup`, which computes the following

.. math::

    \frac{CC\left(f, \frac{g \cdot m - \overline{g \cdot m}}{\sigma_{g \cdot m}}\right)}
         {N_m \cdot \sqrt{\frac{CC(f^2, m)}{N_m} - \left(\frac{CC(f, m)}{N_m}\right)^2}},


where

.. math::

    CC(f,g) = \mathcal{F}^{-1}(\mathcal{F}(f) \cdot \mathcal{F}(g)^*),
    \label{eq:CC}

with :math:`\mathcal{F}` and :math:`\mathcal{F}^{-1}` denoting the forward and inverse Fourier transform, :math:`^*` the complex conjugate, and :math:`\cdot` the element-wise product. :math:`f`, :math:`g` are the target and template, respectively, and :math:`m` is the mask. :math:`N_m` is the number of non-zero elements in :math:`m`, :math:`\overline{g \cdot m}` and :math:`\sigma_{g \cdot m}` the mean and standard deviation of :math:`g \cdot m`, respectively.

``match_template.py`` will compute a bounding box that encapsulates all possible rotations of the template and move the template's center of mass to the center of that box. The computed bounding box will be used as default template mask, if none is specified by the user. Subsequently, ``match_template.py`` will compute the similarity between template and target for all integer-valued voxel translations of the template, together with a range of rotations of the template. The fineness of the rotational grid can be specified by the user.


Configuration
=============

The following provides a background on some of the parameters of ``match_template.py``. For simplicity, the arguments were characterized based on whether they relate to the input, performance or accuracy of ``match_template.py``. The remaining parameters are outlined in the :ref:`command-line-options`.

Input
-----

- **target/template**: Specifies the paths to your target and template files, which are essential inputs for `match_template`.

- **target_mask/template_mask**: Define masks for the target and template, allowing the tool to focus only on regions of interest and potentially improve accuracy.

- **invert_target_contrast**: Whether to invert the target's contrast and perform linear rescaling between zero and one. This option is intended for targets, where the object to-be-matched has negative values, i.e. tomograms with black white contrast.

Performance
-----------

- **cutoff_target/cutoff_template**: By defining a minimal enclosing box containing all significant density values, the execution time can be significantly improved.

- **no_edge_padding**: By default, each dimension of the target is padded by its mean using the template shape. This helps to avoid erroneous scores at the array boundaries, if the input target has to be split in order to fit into memory. If the target does not have to be split its always save to set this option. If the target has to be split, deactivating edge padding speeds up computations, but might require further consideration in postprocessing.

- **no_fourier_padding**: While zero-padding improves numerical stability, avoiding it can lead to faster computations. When this flag is set, the inputs will not be padded to the full convolution shape, which is defined as cumulative box size of template and target minus one. Users might occasionally observe inaccuracies with this option turned.

- **no_centering**: ``match_template.py`` will automatically move the template's center of mass to the center of a new box that can contain all possible rotations of the template around the center of mass. If this flag is set, ``match_template.py`` will not perform this action.

- **interpolation_order**: Defines the spline interpolation's granularity during rotations. While higher values ensure more accuracy reducing it can lead to performance improvements, especially on CPU, at a slight accuracy trade-off.

- **use_mixed_precision**: By utilizing float16 for GPU operations, memory consumption is lowered, and certain hardware might observe a performance boost.

- **use_memmap**: When working with large inputs, this option lets the tool offload objects to the disk, making computations feasible even with limited RAM, though this comes with a slight IO performance overhead.

Accuracy
--------

- **score**: The choice of the scoring function plays a pivotal role in the accuracy of results. Users can select from a variety of options, as detailed in the section :doc:`exhaustive template matching <../reference/matching_exhaustive>`. Note, that if your mask is not symmetric nor encompasses all possible rotations of the template you have to use a scoring method that also rotates the mask, such as FLCF (:py:meth:`tme.matching_exhaustive.flc_scoring`) or MCC (:py:meth:`tme.matching_exhaustive.mcc_scoring`).

- **angular_sampling**: Determines the granularity of the angular sampling for fitting. A finer sampling (lower value) will be more exhaustive, potentially yielding more accurate results but at a computational cost.

- **scramble_phases**: This option scrambles the phases of the templates to simulate a noise background, aiding in refining the score space.

Computation
-----------

- **memory_scaling**: Fraction of available memory that will be used. Corresponds to total amount of RAM for CPU backends, and sum of available GPU memory for GPU backends.

- **ram**: Memory in bytes that should be used for computations. If set will overwrite the value determined by **memory_scaling**.

- **use_gpu**: Wheter to perform computations using a GPU backend.

- **gpu_indices**: GPU device indices to use for computation, e.g. 0,1. If this parameter is provided, the shell environment CUDA_VISIBLE_DEVICES that otherwise specifies the GPU device indices will be ignored.


.. _match-template-usage-examples:

Usage Examples
==============

The following outlines some typical use cases of ``match_template.py``.

.. tab-set::

    .. tab-item:: Particle Picking

        When performing particle picking, the goal is to identify and isolate specific particles (in this case, ribosomes) from a larger dataset (tomogram). By using a known structure as a template, ``match_template.py`` can scan the tomogram to find locations where this structure is likely to appear.

        **Data sources**

        You can download the **target** EMPIAR 10988 by clicking `here <https://www.ebi.ac.uk/empiar/EMPIAR-10988/>`_. Locate the section titled "2. Reconstructed cryo-electron tomograms acquired with defocus-only (DEF) on S. pombe cryo-FIB lamellae". Manually download the file named `TS_037.rec`. It will be in a ZIP archive. Once downloaded, unzip the file to access the tomogram.

        The **template** used to pick ribosomes is EMD-3228, which you can download by clicking `here <https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-3228/map/emd_3228.map.gz>`_. Alternatively, you can use wget:

        .. code-block:: bash

            wget https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-3228/map/emd_3228.map.gz

        **Procedure**

        The following outlines the minimal requirements for using ``match_template.py`` to match ribosomes in a tomogram. Executing the code below will perform template matching using an uniformative template mask, 24 Euler angles, 4 CPU cores, edge -and Fourier padding, without filtering the template and automatically resampling the template to the sampling rate of the tomogram.

        .. code-block:: bash

            match_template.py \
                -m TS_037.rec \
                -i emd_3228.map.gz \
                --invert_target_contrast

        As we have seen in the :doc:`Preprocessing <preprocessing>` guide, template matching accuracy can benefit from the usage of masks to emphasise paticular components of the template and the application of filters to the data. In order to design a suitable mask, we first resample the template to the sampling rate of the tomogram using :py:meth:`tme.density.Density.resample` from within python.

        .. code-block:: python

            from tme import Density

            target = Density.from_file("TS_037.rec", use_memmap = True)
            template = Density.from_file("emd_3228.map.gz")

            template_resampled = template.resample(target.sampling_rate, order = 3)
            template_resampled.to_file("emd_3228_bin4.mrc")

            template_resampled
            # Origin: (0.0, 0.0, 0.0), sampling_rate: (13.481, 13.481, 13.481), Shape: (32, 32, 32)

        Subsequently we can design a spherical mask for template matching. Note that you can just as well use ``emd_3228_bin4.mrc`` in the :ref:`GUI <filter-application>`, but we will use the API here to create a spherical mask centered at the template's center of mass 15, 15, 16, with radius 12 voxel.

        .. code-block:: python

            import numpy as np

            from tme.matching_utils import create_mask

            out = template_resampled.empty
            center = template_resampled.center_of_mass(template_resampled.data).astype(int)
            out.data = create_mask(
                mask_type = "ellipse",
                radius = 12,
                center = center,
                shape = template_resampled.shape,
            )
            out.to_file("emd_3228_bin4_mask.mrc")

            center
            # array([15, 15, 16])

        Due to steric hindrance during data acqusition, the Fourier power spectrum of tomograms contains empty regions, which form a missing wedge for single tilt axis and more complex geometries otherwise. ``match_template.py`` supports applying wedge masks for template matching. However, do note that wedge masks rarely improve template matching accuracy drastically, as they are primarily focused on removing artifacts in order to primarily retain the data that was originally measured. Assuming your data is defined in the xy plane and was tilted over the x axis in 5° steps from -50° to 40°, you can define a wedge mask as follows:

        .. code-block:: bash

            match_template.py \
                -m TS_037.rec \
                -i emd_3228.map.gz \
                --invert_target_contrast \
                --tilt_range 50,40 \
                --wedge_axes 0,2 \
                --tilt_step 5

        .. note::

            The tilt angle in |project| is defined as elevation of the data plane. This differs from the definition some microscopes use. To translate the angle definition, simply subtract the absolute value of the angle from 90°. If you are unsure, feel free to validate the wedge definition using the :ref:`GUI <filter-application>`.


    .. tab-item:: Fit Atomic Structures

        When fitting an atomic structure, the objective is to position a structure or a structure subunit within another density. This method determines the optimal orientation and positioning of the atomic structure within the given density.

        **Data sources**

        You can download the **target** EMD-8621 by clicking `here <https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-8621/map/emd_8621.map.gz>`_. You can download the **template** PDB-5UZ4 by clicking `here <https://files.rcsb.org/download/5UZ4.pdb>`_. Alternatively, you can acquire both using wget:

        .. code-block:: bash

            wget https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-8621/map/emd_8621.map.gz
            wget https://files.rcsb.org/download/5UZ4.pdb


        **Procedure**

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


    .. tab-item:: Alignment of Densities

        For density alignment, the aim is to find the best fit between two densities. This involves determining the optimal orientation and alignment that allows one density to closely match and overlap with another.

        **Data source**

        For this example we will use EMD-8621, which you can download by clicking `here <https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-8621/map/emd_8621.map.gz>`_. Alternatively, you can use wget:

        .. code-block:: bash

            wget https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-8621/map/emd_8621.map.gz


        **Procedure**

        First lets simulate a translation and rotation of the map:

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


Cluster Execution
=================

The efficient memory usage and favorable runtime performance make |project| a great software for high-performance computing environments [1]_. However, please note that multi-node jobs using protocols like `Open MPI <https://www.open-mpi.org/>`_ are not yet supported.

The following subsections provide templates for the execution of |project| on different compute cluster architectures. Since ``match_template.py`` performs the ressource intensive template matching operation, the examples will be in reference to this script.

.. note::

    The ``estimate_ram_usage.py`` script computes an initial memory estimate on a given template matching case for clusters that require memory reservations.

    .. code-block:: bash

        estimate_ram_usage.py --help

SLURM (EMBL)
------------

The scripts in this section are designed in particular for use with the `EMBL <https://www.embl.org/>`_ SLURM cluster. However, they can be adapted to general SLURM clusters by changing the particion name via ``#SBATCH -p htc-el8``.

The templates below are intended to be adapted to your specific use cases and pasted into a new file. This file created by you, lets assume its called ``submission_script.sbatch``, can then be run on the cluster like so:

.. code-block:: bash

    sbatch submission_script.sbatch

Instead of using conda, you can also use the |project| module installed on the cluster. To do so, first determine the newest available version, indicated by it having the higest version number:

.. code-block:: bash

    module spider pyTME

    # Versions:
    #   pyTME/0.1.4-foss-2023a-CUDA-12.1.1
    #   pyTME/0.1.5-foss-2023a

You can then replace every ocurence of:

.. code-block:: bash

    module load Anaconda3
    source activate pytme

by:

.. code-block:: bash

    module load pyTME/0.1.5-foss-2023a

CPU Execution
^^^^^^^^^^^^^

The sbatch file below is sufficient to perform template matching on a bin 4 tomogram with rough shape 500, 900, 900. This job will request 10 CPU cores and 150 GB of RAM for 16 hours on the partition htc-el8. Make sure to replace all variables in curly brackets with your local paths to use this template.

.. code-block:: bash

    #!/bin/bash

    #SBATCH -N 1
    #SBATCH -p htc-el8

    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=10
    #SBATCH --mem 150G

    #SBATCH -t 16:00:00

    module load Anaconda3
    source activate pytme

    match_template.py \
        -m {target_path} \
        -i {template_path} \
        --template_mask {template_mask_path} \
        --cutoff_template 0 \
        -s FLCSphericalMask \
        -n 10 \
        -a {angular_sampling} \
        --no_fourier_padding \
        -o {output_path}

GPU Execution
^^^^^^^^^^^^^

The following will perform template matching using 2 NVIDIA RTX3090 GPUs. If N is the number of GPUs, you can change it by modifying ``#SBATCH --gres=gpu:N`` to a number of your liking.

.. code-block:: bash

    #!/bin/bash
    #SBATCH -N 1
    #SBATCH --export=NONE
    #SBATCH -p gpu-el8

    #SBATCH --mem 100G
    #SBATCH -C gpu=3090
    #SBATCH --gres=gpu:2

    #SBATCH -t 0:30:00

    module load Anaconda3
    source activate pytme

    match_template.py \
        -m $TOMOGRAM_PATH \
        -i $TOMOGRAM_TEMPLATE \
        -n 1 \
        --use_gpu \
        -a $ANGULAR_SAMPLING \
        --no_edge_padding \
        --no_fourier_padding \
        -o bin4.pickle

.. note::

    Using ``--no_edge_padding`` has no impact on the template matching output if the GPU device has sufficient memory, so that the target does not have to be split into smaller chunks. Otherwise, you might experience boundary effects around the splits.


Conclusion
==========

Following this tutorial, you have gained an understanding of template matching using |project| within the realm of electron microscopy. Specifically:

- **Applications**: We explored how template matching functions in diverse scenarios like particle picking, atomic structure fitting, and density alignments.

- **Tool Configuration**: By diving into the `match_template` configurations, we highlighted the balance between accuracy, performance, and input requirements.

- **Practical Usage**: Through specific examples, we showcased how to apply the tool effectively.

In the subsequent tutorial, you will learn how to interpret the output of match_template to extract actionable insigts and optimize the extraction of meaningful information from your electron microscopy data.

.. _command-line-options:

Command-Line Options
====================

.. program:: match_template.py

.. option:: -h, --help

    Show this help message and exit.

.. option:: -m TARGET, --target TARGET

    Path to a target in CCP4/MRC format.

.. option:: --target_mask TARGET_MASK

    Path to a mask for the target target in CCP4/MRC format.

.. option:: --cutoff_target CUTOFF_TARGET

    Target contour level (used for cropping).

.. option:: --cutoff_template CUTOFF_TEMPLATE

    Template contour level (used for cropping).

.. option:: --no_centering

    If set, assumes the template is centered and omits centering.

.. option:: -i TEMPLATE, --template TEMPLATE

    Path to a template in PDB/MMCIF or CCP4/MRC format.

.. option:: --template_mask TEMPLATE_MASK

    Path to a mask for the template in CCP4/MRC format.

.. option:: -o OUTPUT

    Path to output pickle file.

.. option:: -s {CC,LCC,CORR,CAM,FLCSphericalMask,FLC,MCC}

    Template matching scoring function.

.. option:: -n CORES

    Number of cores used for template matching.

.. option:: -r MEMORY, --ram MEMORY

    Amount of memory that can be used in bytes.

.. option:: --memory_scaling MEMORY_SCALING

    Fraction of available memory that can be used. Defaults to 0.85. Ignored if --ram is set.

.. option:: -a ANGULAR_SAMPLING

    Angular sampling rate for template matching. A lower number yields more rotations.

.. option:: -p

    When set, perform peak calling instead of score aggregation.

.. option:: --use_gpu

    Whether to perform computations on the GPU.

.. option:: --gpu_indices GPU_INDICES

    Comma-separated list of GPU indices to use. For example, 0,1 for the first and second GPU. Only used if --use_gpu is set.
    If not provided but --use_gpu is set, CUDA_VISIBLE_DEVICES will be respected.

.. option:: --invert_target_contrast

    Invert the target contrast via multiplication with negative one and linear rescaling between zero and one.
    Note that this might lead to different baseline scores of individual target splits when using unnormalized scores.
    This option is intended for targets, where the object to-be-matched has negative values, i.e. tomograms.

.. option:: --no_edge_padding

    Whether to pad the edges of the target. This is useful, if the target has a well-defined bounding box, e.g. a density map.

.. option:: --no_fourier_padding

    Whether input arrays should be zero-padded to the full convolution shape for numerical stability.
    When working with very large targets such as tomograms it is safe to use this flag and benefit from the performance gain.

.. option:: --scramble_phases

    Whether to phase scramble the template for subsequent normalization.

.. option:: --interpolation_order INTERPOLATION_ORDER

    Spline interpolation used during rotations. If less than zero no interpolation is performed.

.. option:: --use_mixed_precision

    Use float16 for real values operations where possible.

.. option:: --use_memmap

    Use memmaps to offload large data objects to disk. This is particularly useful for large inputs when using --use_gpu.

.. option:: --temp_directory TEMP_DIRECTORY

    Directory for temporary objects. Disks with faster I/O typically improves runtime. Especially in conjunction with **use_memmap**.

.. option:: --gaussian_sigma GAUSSIAN_SIGMA

    Sigma parameter for Gaussian filtering the template.

.. option:: --bandpass_band BANDPASS_BAND

    Comma-separated start and stop frequency for bandpass filtering the template, e.g. 0.1,0.5.

.. option:: --bandpass_smooth BANDPASS_SMOOTH

    Smooth parameter for the bandpass filter.

.. option:: --tilt_range TILT_RANGE

    Comma-separated start and stop stage tilt angle, e.g. 50,45. Used to create a wedge mask to be applied to the template.

.. option:: --tilt_step TILT_STEP

    Step size between tilts, e.g. 5. When set a more accurate wedge mask will be computed.

.. option:: --wedge_axes WEDGE_AXES

    Axis index of wedge opening and tilt axis, e.g. 0,2 for a wedge that is open in z and tilted over x.

.. option:: --wedge_smooth WEDGE_SMOOTH

    Gaussian sigma used to smooth the wedge mask.

References
==========

.. [1] Maurer, V. J.; Siggel, M.; Kosinski, J. PyTME (Python Template Matching Engine): A fast, flexible, and multi-purpose template matching library for cryogenic electron microscopy data. bioRxiv 2023
