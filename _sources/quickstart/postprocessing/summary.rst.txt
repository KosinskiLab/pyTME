.. include:: ../../substitutions.rst

=======
Summary
=======

The ``postprocess.py`` tool analyzes the output of ``match_template.py``.

.. code-block:: bash

    postprocess.py --help

Depending on the downstream use case, different ``output-format`` options are available and outlined below. See the :ref:`coordinate-system` section for details on our coordinate convention.

.. tab-set::

    .. tab-item:: Orientations

        Create a tab-separated file containing x, y, z translations, euler_x, euler_y and euler_z Euler angles, corresponding score and peak caller details.

        .. code-block:: bash

            postprocess.py \
                --input-file output.pickle \
                --output-format orientations \
                --mask-edges \
                --min-boundary-distance 20 \
                --num-peaks 100

    .. tab-item:: Relion 4 / 5

        Create a `STAR <https://en.wikipedia.org/wiki/Self-defining_Text_Archive_and_Retrieval>`_ file compatible with `RELION <https://github.com/3dem/relion>`_ 4 and 5. Relion 4 uses voxel coordinates, Relion 5 centers coordinates and scales them by the voxel size. The version difference is reflected in the header.

        .. code-block:: bash

            postprocess.py \
                --input-file output.pickle \
                --output-format relion4 \
                --mask-edges \
                --min-boundary-distance 20 \
                --num-peaks 100

    .. tab-item:: Alignments

        Transform template into identified orientations and write the output to disk as ``{output_prefix}_{index}.{extension}``. Index 0 corresponds to the top-scoring peak. This format is useful for assessing the result of fitting templates to maps.

        .. code-block:: bash

            postprocess.py \
                --input-file output.pickle \
                --output-format alignment \
                --mask-edges \
                --min-boundary-distance 20 \
                --num-peaks 10

    .. tab-item:: Extraction

        Extract subvolumes around identified peaks and write the output to disk as ``{output_prefix}_{index}.{extension}``. Index 0 corresponds to the top-scoring peak. This format is useful for assessing the result of particle picking.

        .. code-block:: bash

            postprocess.py \
                --input-file output.pickle \
                --output-format extraction \
                --mask-edges \
                --min-boundary-distance 20 \
                --num-peaks 100

    .. tab-item:: Average

        Compute an average of subvolumes based on identified peaks.

        .. code-block:: bash

            postprocess.py \
                --input-file output.pickle \
                --output-format average \
                --mask-edges \
                --min-boundary-distance 20 \
                --num-peaks 100


    .. tab-item:: Pickle

        Create a new pickle file incorporating input files and backgrounds. The output can be used for visual assessment of the normalization procedure and be reused for ``postprocess.py``. The created pickle file can not be used to distinguish between different entities presented by different input files.

        .. code-block:: bash

            postprocess.py \
                --input-file output.pickle \
                --output-prefix output_norm \
                --output-format pickle \
                --background-file background1.pickle background2.pickle

.. tip::

    From version 0.3.0 onwards, postprocessing supports advanced multi-input and background correction. Multiple input files can be specified to distinguish between different macromolecular species, with corresponding class identifiers made available in orientations and RELION output formats. Additionally, multiple background corrections can be applied simultaneously via ``--background-file``, enabling users to account for various noise sources beyond single backgrounds (e.g., from ``--scramble-phases``) and incorporate complex cellular environments such as membrane backgrounds.
