.. include:: ../../substitutions.rst

=================
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
    #   pyTME/0.2.0-foss-2023a-CUDA-12.1.1
    #   pyTME/0.2.1-foss-2023a-CUDA-12.1.1

You can then replace every ocurence of:

.. code-block:: bash

    module load Anaconda3
    source activate pytme

by:

.. code-block:: bash

    module pyTME/0.2.1-foss-2023a-CUDA-12.1.1

CPU Execution
-------------

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
        -m ${TOMOGRAM_PATH} \
        -i ${TEMPLATE_PATH} \
        --template_mask ${TEMPLATE_MASK_PATH} \
        -n 10 \
        -a $ANGULAR_SAMPLING \
        -o ${OUTPUT_PATH}


GPU Execution
-------------

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
        -m ${TOMOGRAM_PATH} \
        -i ${TEMPLATE_PATH} \
        --template_mask ${TEMPLATE_MASK_PATH} \
        -n 2 \
        -a $ANGULAR_SAMPLING \
        -o ${OUTPUT_PATH} \
        --use_gpu


References
----------

.. [1] Maurer, V. J.; Siggel, M.; Kosinski, J. PyTME (Python Template Matching Engine): A fast, flexible, and multi-purpose template matching library for cryogenic electron microscopy data. SoftwareX 2024
