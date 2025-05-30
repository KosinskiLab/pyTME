.. include:: ../../substitutions.rst
.. currentmodule:: tme.matching_optimization

Non-exhaustive
==============

Non-exhaustive template matching aims to identify a global optimum without evaluating all possible configurations. Whether non-exhaustive template matching identifies a global optimum is highly dependent on the topology of the score space and available constraints. Typically, non-exhaustive template matching is useful when the rough orientation is known.

Within |project|, non-exhaustive template matching problems are distinguished based on whether they operate on :obj:`numpy.ndarray` objects or coordinates. Therefore, we categorize them into three different classes of matching problems:

.. tab-set::

   .. tab-item:: Density to Density

      Analogous to exhaustive template matching operations in |project|, the classes below identify the configuration maximizing the similarity
      between two densities.

      .. autosummary::
         :toctree: ../api/

         FLC


   .. tab-item:: Coordinates to Density

      The classes below maximize the similarity between a coordinate set with associated weights and a density. This case is commonly found when matching atomic structures to electron density maps.

      .. autosummary::
         :toctree: ../api/

         CrossCorrelation
         LaplaceCrossCorrelation
         NormalizedCrossCorrelation
         NormalizedCrossCorrelationMean
         MaskedCrossCorrelation
         PartialLeastSquareDifference
         Chamfer
         MutualInformation
         Envelope
         NormalVectorScore


   .. tab-item:: Coordinates to Coordinates

      The classes below maximize the similarity between two coordinate sets and associated weights. This could be special points within densities, regular coordinate sets, or vectors between template and target that should be aligned.

      .. autosummary::
         :toctree: ../api/

         Chamfer
         NormalVectorScore

In order to perform non-exhaustive template matching, we have to initialize one of the objects outlined above. The following will showcase :py:class:`CrossCorrelation`, but the procedure can be extended analogously to all scores shown above. We can use __call__ method of the created ``score_object`` to compute the score of the current configuration.

.. code-block:: python

   import numpy as np
   from tme.matching_utils import create_mask
   from tme.matching_optimization import CrossCorrelation

   target = create_mask(
      mask_type="ellipse",
      radius=(15,10,5),
      shape=(51,51,51),
      center=(25,25,25),
   )

   # Note this could also be any other template that can be expressed as
   # coordinates, e.g. an atomic structure.
   template = create_mask(
      mask_type="ellipse",
      radius=(5,10,15),
      shape=(51,51,51),
      center=(32,17,25),
   )
   template_coordinates = np.array(np.where(template > 0))
   template_weights = template[tuple(template_coordinates)]

   score_object = CrossCorrelation(
      target=target,
      template_coordinates=template_coordinates,
      template_weights=template_weights,
      negate_score=True # Multiply returned score with -1 for minimization
   )

   score_object() # -543.0

Alternatively, ``score_object`` could have been instantiated using the wrapper function :py:meth:`create_score_object`.

.. autosummary::
   :toctree: ../api/

   create_score_object


:py:meth:`optimize_match` provides routines for non-exhaustive template matching given a ``score_object`` and bounds on possible translations and rotations.

.. autosummary::
   :toctree: ../api/

   optimize_match

Given our previously defined ``score_object``, we can find the optimal configuration:

.. code-block:: python

   from tme.matching_optimization import optimize_match

   translation, rotation, refined_score = optimize_match(
      score_object=score_object,
      optimization_method="basinhopping",
      maxiter=50
   )
   translation
   # array([-7.04120465,  8.17709819,  0.07964882])
   rotation
   # array([[ 3.0803247e-03,  1.7386347e-04,  9.9999523e-01],
   #        [ 3.9309423e-02,  9.9922705e-01, -2.9481627e-04],
   #        [-9.9922234e-01,  3.9310146e-02,  3.0711093e-03]],
   #  dtype=float32)
   refined_score #-3069.0

.. note::

   :py:meth:`optimize_match` will update ``score_object`` so using __call__ afterwards will return ``refined_score``. Smaller values of `maxiter` might yield suboptimal results.

The computed ``translation`` and ``rotation`` can then be used to recover the configuration of the template with maximal similarity to the target. This is functionally similar to the operation performed using :py:meth:`Density.match_densities <tme.density.Density.match_densities>` and :py:meth:`Density.match_structure_to_density <tme.density.Density.match_structure_to_density>`.


.. code-block:: python

   from tme import Density

   template_density = Density(template.astype(np.float32))
   template_density = template_density.rigid_transform(
      translation=translation,
      rotation_matrix=rotation,
      order=3,
      use_geometric_center=True
   )

:py:meth:`optimize_match` offers various optimization schemes to balance accuracy and runtime efficiency. Here we used ``optimization_method="basinhopping"``, which yields a good balance between those factors. For performance-critical applications, ``optimization_method="minimize"`` in conjunction with :py:class:`CrossCorrelation.grad` can be used, which incorporates the analytical gradient of the scoring function into the optimization process. This approach can significantly speed up convergence, especially when the initial configuration is close to the global optimum.

The example below highlights this key characteristic. The initial orientation is too far from the correct one to be recovered without better initial values or tighter bounds on translation and rotation. Consequently, the optimization scheme becomes trapped in a local optimum. In practice, multiple initial conditions and potentially bounds should be evaluated to identify the global optimum.

.. code-block:: python

   score_object = CrossCorrelation(
      target=target,
      template_coordinates=template_coordinates,
      template_weights=template_weights,
      negate_score=True,
      return_gradient=True
   )

   translation, rotation, refined_score = optimize_match(
      score_object=score_object,
      optimization_method="minimize",
      maxiter=500,
      x0=[0,0,0,0,50,0]
   )
   translation
   # array([-4.51057473,  9.10106869,  1.40713668])
   rotation
   # array([[ 0.09346937,  0.00730731,  0.99559534],
   #       [-0.08880167,  0.9960488 ,  0.00102632],
   #       [-0.99165404, -0.08850646,  0.09374895]],
   #  dtype=float32)
   refined_score #-2254.263671875

.. tip::

   For custom optimization strategies, developers can utilize the :py:meth:`CrossCorrelation.score` method, which accepts a tuple of translations and Euler angles as input.