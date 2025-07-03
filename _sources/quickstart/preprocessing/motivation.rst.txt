==========
Motivation
==========

Template matching relies heavily on the quality of input data and commonly yields erroneous matches should the quality be insufficient. Through preprocessing, we can enhance the similarity between the template and its instances in the target, in turn improving recall and precision.


Background
----------

In order to understand which factors to focus on during preprocessing, we have to take a schematic look at the computation of the normalized cross-correlation

.. math::

   \frac{(x - \mu)}{\sigma}.

- :math:`x` represents the non-normalized cross-correlation between the target and the template.

- :math:`\mu` represents the non-normalized cross-correlation between the target and a mask.

- :math:`\sigma` represents the variance in non-normalized cross-correlation between the target and a mask.


Based on this schematic assessment, we can improve at the stage of template matching through two principal approaches

1. Filtering to improve the similarity between template and template instances in the target.

2. Design a mask that provides a faithful approximation of the cross-correlation background given the template.

These preprocessing approaches will be discussed in the following sections. For a mathematically more rigorous treatment of the normalized cross-correlation we refer the reader to [1]_.

References
----------

.. [1] Briechle, K.; Hanebeck, U. D. Template matching using fast normalized cross correlation. Optical Pattern Recognition XII. 2001; pp 95–102.
