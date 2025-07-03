.. include:: ../../substitutions.rst

====
IMOD
====

In the case of particle picking its usually sufficient to look at the translations in a viewer like `IMOD <https://bio3d.colorado.edu/imod/>`_. The following assumes that you have IMOD installed and its command line tools linked.

.. code-block:: bash

    awk -F'\t' '
        BEGIN {OFS="\t"}
        NR==1 {next}
        {print 1, 1, $3, $2, $1}
    ' output.tsv > coordinates.tsv

    point2model -inp coordinates.tsv -ou coordinates.mod -ci 10
