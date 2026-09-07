.. include:: ../substitutions.rst

.. _coordinate-system:

===========
Conventions
===========

This page collects conventions used throughout |project|. It covers the
coordinate system, the Euler-angle convention, and the layout of pickle files
emitted by ``pytme match``.

Coordinate System
-----------------

Our convention follows the schematics outlined in [1]_. We use a right-handed
coordinate system with orthogonal X, Y, and Z axes. Euler angles are expressed
using the intrinsic ZYZ convention, with the first rotation around the Z axis,
the second around the new Y axis, and the third around the new Z axis (see
:py:meth:`euler_to_rotationmatrix <tme.rotations.euler_to_rotationmatrix>`).
The default orientation is the z-unit vector (0, 0, 1).

Pickle File Layout
------------------

The output of ``pytme match`` is a `pickle
<https://docs.python.org/3/library/pickle.html>`_ file containing a tuple. All
but the last element correspond to the return value of a given
:doc:`analyzer </reference/analyzer/base>` merge method. The file can be read
using :py:meth:`load_pickle <tme.matching_utils.load_pickle>`.

For the default analyzer :py:class:`MaxScoreOverRotations
<tme.analyzer.MaxScoreOverRotations>` the pickle file contains

- **Scores**: score for each position in the target.
- **Offset**: coordinate system shift.
- **Rotations**: optimal rotation index for each translation.
- **Rotation Dictionary**: mapping from rotation indices to rotation matrices.
- **Sum of Squares**: sum of squares of scores for statistics.
- **Metadata**: coordinate system information and parameters for reproducibility.

When the ``-p`` flag is passed to ``pytme match`` the output structure differs

- **Translations**: peak position.
- **Rotations**: rotation matrix describing template orientation at peak.
- **Scores**: score at peak.
- **Details**: additional properties of the peak.
- **Metadata**: coordinate system information and parameters for reproducibility.

References
----------

.. [1] Heymann, J.B.; Chagoyen, M.; Belnap, D.M. Common conventions for
       interchange and archiving of three-dimensional electron microscopy
       information in structural biology. J Struct Biol 2005, 151, 196-207.
