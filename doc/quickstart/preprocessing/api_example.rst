.. include:: ../../substitutions.rst

=============
Using the API
=============

The :py:class:`Preprocessor <tme.preprocessor.Preprocessor>`, class defines a range of filters that are outlined in the API reference. It's methods operates on :obj:`numpy.ndarray` instances, such as the :py:attr:`data <tme.density.Density.data>` of :py:class:`Density <tme.density.Density>` instances. The :py:class:`create_mask <tme.matching_utils.create_mask>` utility on the other hand can be used to define a variety of masks for template matching.

The following outlines the creation of an ellipsoidal mask using :py:class:`create_mask <tme.matching_utils.create_mask>` to create an ellipsoid, in this case a sphere, with radius 15 as mask. :py:meth:`create_mask <tme.matching_utils.create_mask>` returns a :obj:`numpy.ndarray` object, which we use in turn to create a :py:class:`Density <tme.density.Density>` instance. Note that by default, the instances attributes :py:attr:`sampling_rate <tme.density.Density.sampling_rate>` will be initialized to one unit per voxel, and the :py:attr:`origin <tme.density.Density.origin>` attribute to zero. We can write the created mask to disk using :py:meth:`Density.to_file <tme.density.Density.to_file>`.


.. code-block:: python

	from tme import Density
	from tme.matching_utils import create_mask

	mask = create_mask(
		mask_type = "ellipse",
		center = (25, 25, 25),
		shape = (50, 50, 50),
		radius = (15, 15, 15),
		sigma_decay = 2
	)
	Density(mask).to_file("example_gaussian_filter.mrc")

You can visualize the output using your favourite macromolecular viewer. For simplicity, we are showing a maximum intensity projection. The left hand side shows the unfiltered ``example.mrc``, while the right hand side displays the filtered ``example_gaussian_filter.mrc``. The degree of smoothing depends on the value of the ``sigma`` parameter.

.. plot::
 	:caption: Mask creation and filter application using the API.

	import copy

	import numpy as np
	import matplotlib.pyplot as plt
	import matplotlib.colors as colors

	from tme import Density, Preprocessor
	from tme.matching_utils import create_mask

	preprocessor = Preprocessor()
	mask = create_mask(
		mask_type = "ellipse",
		center = (25, 25, 25),
		shape = (50, 50, 50),
		radius = (15, 15, 15)
	).astype(np.float32)

	filtered_mask = preprocessor.gaussian_filter(
		template = mask, sigma = 2
	)

	mask = mask.max(axis = 0)
	filtered_mask = filtered_mask.max(axis = 0)

	fig, axs = plt.subplots(
		nrows=1, ncols=2, sharex=False, sharey=False, figsize=(12, 5)
	)
	colormap = copy.copy(plt.cm.viridis)
	colormap.set_bad(color="white", alpha=0)
	norm = colors.Normalize(vmin=0, vmax=1)

	mask[mask < np.exp(-2)] = np.nan
	filtered_mask[filtered_mask < np.exp(-2)] = np.nan

	axs[0].imshow(mask, cmap=colormap, norm = norm)
	axs[1].imshow(filtered_mask, cmap=colormap, norm = norm)
	axs[0].set_title("Spherical Mask", color="#24a9bb")
	axs[1].set_title("Spherical Mask + Gaussian Filter", color="#24a9bb")

The :py:class:`Preprocessor <tme.preprocessor.Preprocessor>`, supports a range of other operations such as the creation of wedge masks. The following outlines the creation of a continuous wedge mask assuming an infinite plane using :py:meth:`continuous_wedge_mask <tme.preprocessor.Preprocessor.continuous_wedge_mask>`.


.. code-block:: python

	import numpy as np
	from tme import Density, Preprocessor

	preprocessor = Preprocessor()

	mask = preprocessor.continuous_wedge_mask(
		shape = (32,32,32),
		opening_axis = 0,
		tilt_axis = 2,
		start_tilt = 40,
		stop_tilt = 40,
		# RELION assumes symmetric FFT
		omit_negative_frequencies = False
	)

	Density(mask).to_file("wedge_mask.mrc")

By default, all Fourier operations in |project| assume the DC component to be at the origin of the array. To observe the typical wedge shape you have to shift the DC component of the mask to the array center, for instance by using :obj:`numpy.fft.fftshift` like so:

.. code-block:: python

	import numpy as np

	mask_center = np.fft.fftshift(mask)
	Density(mask_center).to_file("wedge_mask_centered.mrc")