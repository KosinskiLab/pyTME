.. _cli-match_template:

=======
Summary
=======

``match_template.py`` facilitates template matching using various scoring metrics.

.. code-block:: bash

    match_template.py --help

``preprocess.py`` enables template generation, masks can be created interactively using ``preprocessor_gui.py`` or the API.

.. note::

    ``match_template.py`` expects high densities to have positive (white contrast), low densities to have negative values (black contrast). Therefore, input tomograms typically need to be inverted by multiplication with negative one. Version 0.1.4 introduced the ``--invert_target_contrast`` flag, which automatically performs the inversion for you.

    As of version 0.2.1 you can alternatively provide an appropriately scaled template. This template does need to be centered and ``match_template.py`` needs to be invoked with the ``--no_centering`` flag.
