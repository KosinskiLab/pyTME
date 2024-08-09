.. include:: ../../substitutions.rst

==========
Motivation
==========

Template matching offers a powerful approach towards object detection. Unlike most deep learning methods, template matching generalizes to arbitrary templates. However, the generalization capacity comes with complexity. Not only is template matching computationally expensive, it also requires tuning filtering and masking parameter in order to achieve performance comparable to deep learning. Fortunately, the effect of different filters on template matching performance is highly interpretable, enabling us to integrate knowledge of the phyiscal underlyings of the system as well as prior experience.


Background
----------

Template matching is used under various names in the electron microscopy field

- Fitting an atomic structure to a density map.

- Particle picking on tomograms, tilt series, images.

- Alignment in subtomogram averaging.

Despite their naming, they all follow the same template matching formalism. The goal is to optimize the similarity between a template and a target along all translational and rotational degrees of freedom. The majority of software uses a variation of the normalized cross-correlation as similarity metric

.. math::

    \frac{CC\left(f, \frac{g \cdot m - \overline{g \cdot m}}{\sigma_{g \cdot m}}\right)}
         {N_m \cdot \sqrt{\frac{CC(f^2, m)}{N_m} - \left(\frac{CC(f, m)}{N_m}\right)^2}},


where

.. math::

    CC(f,g) = \mathcal{F}^{-1}(\mathcal{F}(f) \cdot \mathcal{F}(g)^*),
    \label{eq:CC}

with :math:`\mathcal{F}` and :math:`\mathcal{F}^{-1}` denoting the forward and inverse Fourier transform, :math:`^*` the complex conjugate, and :math:`\cdot` the element-wise product. :math:`f`, :math:`g` are the target and template, respectively, and :math:`m` is the template mask. :math:`N_m` is the number of non-zero elements in :math:`m`, :math:`\overline{g \cdot m}` and :math:`\sigma_{g \cdot m}` the mean and standard deviation of :math:`g \cdot m`, respectively.

.. note::

	The score shown above corresponds to :py:meth:`flcSphericalMask <tme.matching_scores.flcSphericalMask_setup>`, which is the default score used by ``match_template.py``.