.. include:: ../substitutions.rst

============
Integrations
============

The output of |project| can be readily integrated with the refinement, classification and averaging procedures of existing software.

|project| defines :py:class:`Orientations <tme.orientations.Orientations>`, which can convert the tsv-based format produced with ``postprocess.py`` and ``--output_format orientations``, into a range of :py:meth:`available formats <tme.orientations.Orientations.to_file>`, such as star files. This enables compatibility with a range of software and is achieved as follows


.. code-block:: python

    from tme import Orientations
    orientations = Orientations.from_file("orientations.tsv")
    orientations.to_file("orientations.star", name = "/path/to/your/tomogram")


The following sections are dedicated to integrations with specific software

.. toctree::

   postprocessing/relion
   postprocessing/imod


