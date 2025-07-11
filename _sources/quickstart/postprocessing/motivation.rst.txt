.. include:: ../../substitutions.rst

==========
Motivation
==========

Postprocessing identifies regions of high similarity from the template matching output, which we refer to as peaks. Each peak corresponds to an occurence of the template in the target, which is fully characterized by a translation and rotation matrix. Identifying peaks is challenging by itself, hence |project| implements a variety of analysis strategies for optimal performance.


Background
----------

Lets recall an example from the preprocessing section. The highest peak is located within the red rectangle at 111, 240. Therefore, the maximum similariy is obtained when translating the template so that its center is at position 111, 240. This convention used in |project| to represent in orientations is analogous to other tools and figuratively explained in a `skimage tutorial <https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_template.html>`_. The corresponding rotation matrix for this peak is the identity matrix.

.. plot::

    import copy
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    from tme import Density
    from tme.cli import match_template

    target = Density.from_file("../../_static/examples/preprocessing_target.png").data
    template = Density.from_file("../../_static/examples/preprocessing_template.png").data

    result = match_template(target, template)[0]
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(result)
    ax.set_title('Template Matching Score', color='#0a7d91')

    square_size = max(template.shape)
    rect = patches.Rectangle((x - square_size / 2, y - square_size / 2), square_size, square_size, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    plt.tight_layout()
    plt.show()


Template Matching Output
------------------------

``match_template.py`` evaluates the similarity between target and template along all rotational and translation degrees of freedom. The output is a `pickle <https://docs.python.org/3/library/pickle.html>`_ file, whose content depends on the corresponding analyzer. The file can be read using :py:meth:`load_pickle <tme.matching_utils.load_pickle>`. For the default analyzer :py:class:`MaxScoreOverRotations <tme.analyzer.MaxScoreOverRotations>`, it contains five objects:

- **Scores**: An array with scores mapped to translations.
- **Offset**: Offset informing about shifts in coordinate sytems.
- **Rotations**: An array of optimal rotation indices for each translation.
- **Rotation Dictionary**: Mapping of rotation indices to rotation matrices.
- **Metadata**: Coordinate system information and parameters for reproducibility.

However, when you use the `-p` flag the output structure differs

- **Translations**: A numpy array containing translations of peaks.
- **Rotations**: A numpy array containing rotations of peaks.
- **Scores**: Score of each peak.
- **Details**: Additional information regarding each peak.
- **Metadata**: Coordinate system information and parameters for reproducibility.

.. note::

    In general, all but the last element of the created output pickle will correspond to return value of a given :doc:`analyzer </reference/analyzer/base>`'s merge method.
