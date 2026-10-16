.. currentmodule:: tme.analyzer

Score Aggregation
=================

Score aggregation analyzers accumulate optimal scores and rotation information across the search space. These analyzers maintain the best scoring results for each spatial location or rotation, enabling efficient identification of high-confidence matches.

.. autosummary::
   :toctree: ../api/

   MaxScoreOverRotations
   MaxScoreOverTranslations
   MaxScoreOverRotationsConstrained

**MaxScoreOverRotations** finds the rotation maximizing the score over all possible translations for each spatial position.

**MaxScoreOverTranslations** determines the translation maximizing the score over all possible rotations for each orientation.

**MaxScoreOverRotationsConstrained** enables constrained template matching using rejection sampling with angular constraints and positional priors.
