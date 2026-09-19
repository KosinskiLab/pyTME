.. include:: ../../substitutions.rst

=================
Cluster Execution
=================

This tutorial demonstrates how to scale template matching to large datasets. When processing dozens or hundreds of tomograms, manually creating and submitting individual jobs becomes impractical. ``pytme batch`` automates this workflow by discovering datasets, generating cluster scripts, and managing job submission.

From version 0.3.1 onwards, the runner has two modes

- ``pytme batch matching``: Run template matching
- ``pytme batch analysis``: Analyze template matching results

These can be run independently or as part of a complete pipeline.


Dataset Organization
--------------------

For this tutorial, we extend the :doc:`ribosome picking example <particle_picking>` to a larger dataset. Your project directory will typically look like this

.. code-block:: text

    project_directory/
    ├── tomograms/                      # Tomograms
    │   ├── TS_037_10.00Apx.rec
    │   ├── TS_041_10.00Apx.rec
    │   └── TS_045_10.00Apx.rec
    ├── metadata/                       # Metadata files
    │   ├── TS_037.mdoc                 # Can also be Warp/M XMLs
    │   ├── TS_041.mdoc                 # or tomostar STAR files
    │   └── TS_045.mdoc
    ├── masks/                          # Optional tomogram masks
    │   ├── TS_037_mask.mrc
    │   ├── TS_041_mask.mrc
    │   └── TS_045_mask.mrc
    └── templates/
        ├── emd_3228_resampled.mrc      # 80S ribosome template
        └── emd_3228_resampled_mask.mrc # 80S ribosome mask

The batch runner automatically extracts tomogram identifiers by removing technical suffixes like pixel size information (``_10.00Apx``) and matches files across directories.


Basic Batch Processing
----------------------

The following outlines how to perform a basic template matching and analysis run.

How batch jobs are executed is set by three parameters of ``pytme batch``

1. ``--submit-command`` is the command that submits each generated job
2. ``--submit-args`` carries its CPU, memory, GPU, and time requests
3. ``--environment-setup`` is run before every job to set up an environment that has ``pytme`` installed

By default, ``--submit-command`` is ``sbatch`` (SLURM) and comes with sensible defaults for each operation. Running locally with ``--submit-command bash`` is equally simple, while other schedulers require adapting the submission flags.

Matching
^^^^^^^^

The template matching workflow identifies all tomograms and metadata files using glob patterns

.. code-block:: bash

    pytme batch matching \
        --tomograms "project_directory/tomograms/*.rec" \
        --metadata "project_directory/metadata/*.mdoc" \
        --template templates/emd_3228_resampled.mrc \
        --template-mask templates/emd_3228_resampled_mask.mrc \
        --particle-diameter 300 \
        --output-dir ribosome_batch_001/results \
        --script-dir ribosome_batch_001/scripts \
        --backend cupy \
        --dry-run

.. note::

    The quotation marks are required for parsing of glob patterns. If your tomogram names end with ``.mrc``, you would adapt the glob pattern to ``"project_directory/tomograms/*.mrc"``.

This command will

1. **Discover** all ``.rec`` files in the tomograms directory
2. **Match** each tomogram with its corresponding ``.mdoc`` metadata file
3. **Generate** a submission script for each valid pair

The generated scripts can be submitted manually, or automatically by omitting the ``--dry-run`` flag. With the SLURM defaults, each script follows this pattern

.. code-block:: bash

    #!/bin/bash

    # Environment setup
    module load pyTME

    # Submission command
    # sbatch --job-name=pytme_match_TS_037 --output=ribosome_batch_001/results/TS_037_%j.out \
    #     --error=ribosome_batch_001/results/TS_037_%j.err --ntasks=1 --nodes=1 --ntasks-per-node=1 \
    #     --cpus-per-task=4 --mem=32G --time=05:00:00 --partition=gpu-el8 --qos=normal --export=none \
    #     --gres=gpu:1 ribosome_batch_001/scripts/pytme_TS_037.sh

    pytme match \
        --target project_directory/tomograms/TS_037_10.00Apx.rec \
        --output ribosome_batch_001/results/TS_037_10.00Apx.pickle \
        --ctf-file project_directory/metadata/TS_037.mdoc \
        --tilt-angles project_directory/metadata/TS_037.mdoc \
        --template templates/emd_3228_resampled.mrc \
        --template-mask templates/emd_3228_resampled_mask.mrc \
        --backend cupy \
        --particle-diameter 300

The submission command is recorded as a comment for reference, and the resources following it come straight from ``--submit-args``.

To request different resources, pass ``--submit-args`` explicitly. The placeholders ``{job_name}``, ``{output_file}``, and ``{error_file}`` are filled in per task, while the remaining flags reach the scheduler unchanged. The example below picks a queue and QOS, requests two GPUs, and raises the time limits

.. code-block:: bash

    pytme batch matching \
        --tomograms "project_directory/tomograms/*.rec" \
        --metadata "project_directory/metadata/*.mdoc" \
        --template templates/emd_3228_resampled.mrc \
        --particle-diameter 300 \
        --submit-args "--job-name={job_name} --output={output_file} --error={error_file} --partition=gpu-el8 --qos=high --gres=gpu:2 --cpus-per-task=16 --mem=64G --time=12:00:00" \
        --dry-run

.. note::

    Monitor submitted jobs with your scheduler's tooling. On SLURM, ``squeue --me`` lists your jobs, ``scontrol show job <id>`` shows details, and ``sacct -j <id> --format=JobID,JobName,MaxRSS,Elapsed`` reports resource usage.


Analysis
^^^^^^^^

After template matching completes, use the analysis workflow to identify peaks and generate particle coordinates. The analysis workflow is CPU-only and much faster than template matching.

.. code-block:: bash

    pytme batch analysis \
        --input-files "ribosome_batch_001/results/*.pickle" \
        --num-peaks 1000 \
        --output-format relion4 \
        --output-dir ribosome_batch_001/picks \
        --script-dir ribosome_batch_001/picks_scripts \
        --dry-run

Output
^^^^^^

Results are organized in the following manner

.. code-block:: text

    ribosome_batch_001/
    ├── results/
    │   ├── TS_037_10.00Apx.pickle     # Template matching results
    │   ├── TS_037_12345.out           # SLURM logs
    │   ├── TS_041_10.00Apx.pickle
    │   ├── TS_041_12346.out
    │   ├── TS_045_10.00Apx.pickle
    │   └── TS_045_12347.out
    └── picks/
        ├── TS_037_10.00Apx.star       # Peak coordinates
        ├── TS_037_12345.out           # SLURM logs
        ├── TS_041_10.00Apx.star
        ├── TS_041_12346.out
        ├── TS_045_10.00Apx.star
        └── TS_045_12347.out


Processing Subsets
------------------

To process only specific tomograms, create a list file

.. code-block:: bash

    # Create tomogram selection
    echo "TS_037" > selected_tomos.txt
    echo "TS_041" >> selected_tomos.txt

    # Process only selected tomograms
    pytme batch matching \
        --tomograms "project_directory/tomograms/*.rec" \
        --metadata "project_directory/metadata/*" \
        --template templates/emd_3228_resampled.mrc \
        --tomo-list selected_tomos.txt \
        --particle-diameter 300 \
        --dry-run


Advanced Options
----------------

The following outlines advanced features for production workflows, including filtering, background correction, and multi-entity analysis.

Filtering
^^^^^^^^^

For production runs, you may want to include additional filters similar to those described in the ribosome picking tutorial

.. code-block:: bash

    pytme batch matching \
        --tomograms "project_directory/tomograms/*.rec" \
        --metadata "project_directory/metadata/*.mdoc" \
        --masks "project_directory/masks/*mask.mrc" \
        --template templates/emd_3228_resampled.mrc \
        --template-mask templates/emd_3228_resampled_mask.mrc \
        --particle-diameter 300 \
        --lowpass 40 \
        --tilt-weighting relion \
        --whiten-spectrum \
        --amplitude-contrast 0.08 \
        --spherical-aberration 2.7 \
        --voltage 300 \
        --output-dir results/ribosome_batch_001 \
        --dry-run


Compared to the basic run above, this now includes

- Tomogram masks to exclude problematic regions
- Lowpass filtering to 40 Ångstrom
- Missing wedge correction with RELION-style tilt weighting
- Spectral whitening to enhance weak signals

.. tip::

    You can switch between compute backends via ``--backend``, for instance ``cupy`` or ``pytorch`` for GPU execution.


Mixed Formats
^^^^^^^^^^^^^

You can mix formats by adapting the glob patterns. For instance for metadata

.. code-block:: bash

    pytme batch matching \
        --tomograms "project_directory/tomograms/*.rec" \
        --metadata "project_directory/metadata/*" \
        --template templates/emd_3228_resampled.mrc \
        --particle-diameter 300 \
        --dry-run

The ``metadata/*`` pattern will match ``.mdoc``, ``.xml``, ``.star``, and other supported formats, automatically pairing each tomogram with its corresponding metadata file. However, note that when multiple metadata files exist for a given tomogram, the runner will default to the first one it encountered.


Background Correction
^^^^^^^^^^^^^^^^^^^^^

In some cases, e.g. membrane proteins, it can be helpful to perform matching for templates other than your structure of interest, in order to suppress background peaks and improve detection. As of version 0.3.2 one built-in background correction approach is ``--background-correction phase-scrambling``. For custom templates you can run

.. code-block:: bash

    pytme batch matching \
        --tomograms "project_directory/tomograms/*.rec" \
        --metadata "project_directory/metadata/*.mdoc" \
        --template templates/other_template_resampled.mrc \
        --template-mask templates/emd_3228_resampled_mask.mrc \
        --particle-diameter 300 \
        --output-dir ribosome_batch_001/results_noise \
        --script-dir ribosome_batch_001/scripts_noise \
        --dry-run

Then run analysis with background correction

.. code-block:: bash

    pytme batch analysis \
        --input-files "ribosome_batch_001/results/*.pickle" \
        --background-files "ribosome_batch_001/results_noise/*.pickle" \
        --num-peaks 1000 \
        --output-format relion4 \
        --output-dir ribosome_batch_001/picks_norm \
        --script-dir ribosome_batch_001/picks_scripts \
        --dry-run

Multiple Entities and Backgrounds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The analysis workflow supports combining results from multiple template matching runs. This is useful when distinguishing between different templates

.. code-block:: bash

    pytme batch analysis \
        --input-files "ribosome_batch_001/results/*.pickle" "ribosome_batch_001/results_rnap/*.pickle" \
        --background-files "ribosome_batch_001/results_noise/*.pickle" \
        --num-peaks 1000 \
        --output-format relion4 \
        --output-dir ribosome_batch_001/picks \
        --script-dir ribosome_batch_001/picks_scripts \
        --dry-run

When multiple input patterns are provided, the analysis workflow will

- Aggregate correlation scores from all matching runs for each tomogram
- Take the maximum score at each position across all inputs
- Apply background correction using all provided background datasets
- Generate a single coordinate file per tomogram with peaks and class labels corresponding to the order of input files.

You can also include multiple background datasets for more custom normalization

.. code-block:: bash

    pytme batch analysis \
        --input-files "ribosome_batch_001/results/*.pickle" "ribosome_batch_001/results_rnap/*.pickle" \
        --background-files "ribosome_batch_001/results_noise/*.pickle" "ribosome_batch_001/results_membrane/*.pickle" \
        --num-peaks 1000 \
        --output-format relion4 \
        --output-dir ribosome_batch_001/picks \
        --script-dir ribosome_batch_001/picks_scripts \
        --dry-run
