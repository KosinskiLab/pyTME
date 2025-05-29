:notoc:

.. include:: substitutions.rst

.. module:: |project|

***********************
|project| documentation
***********************

**Date**: |today| **Version**: |release|

|project| is a Python library for data-intensive n-dimensional template matching using CPUs and GPUs.


**Key Features**

- :doc:`Flexible backends <reference/backends>` to run the same code on diverse hardware platforms using a best-of-breed approach.

- :doc:`Analyzer <reference/template_matching/analyzer>` to inject custom code enabling real-time processing and manipulation of template matching scores.

- Specialized tools for cryogenic electron microscopy from CTF correction, :doc:`data structures <reference/data_structures/density>`, a :doc:`GUI <quickstart/preprocessing/gui_example>` for interactive mask creation and filter exploration, to :doc:`integrations <quickstart/integrations>` with other commonly used software.


.. grid:: 2

    .. grid-item-card::
       :img-top: _static/index_getting_started.svg

       User Guide
       ^^^^^^^^^^

       Learn how to use |project|.

       +++

       .. button-ref:: quickstart/index
             :expand:
             :color: secondary
             :click-parent:

             To the user guide

    .. grid-item-card::
       :img-top: _static/index_api.svg

       API reference
       ^^^^^^^^^^^^^

       Detailed documentation of the API.

       +++

       .. button-ref:: reference/index
             :expand:
             :color: secondary
             :click-parent:

             To the reference

.. toctree::
   :maxdepth: 1
   :hidden:
   :titlesonly:

   User Guide <quickstart/index>
   API <reference/index>
   Index <genindex>
