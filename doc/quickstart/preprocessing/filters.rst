.. include:: ../../substitutions.rst

====================
Filtering Techniques
====================

Filtering is a crucial preprocessing step in template matching, primarily serving to remove noise and to emphasize particular components of the data. The following uses 2D images for illustration, but the principles apply equally to other data modalities.


Noise Removal
-------------

Real-world data often contains noise from various sources, including sensor imperfections and environmental factors. This noise can significantly impact template matching accuracy by obscuring true features or introducing false patterns. Filters are instrumental in mitigating the impact of noise. In thef following, we explore the impact of two types of noise, Gaussian and salt-and-pepper on template matching. We then compare the similarity scores with and without the application of appropriate filters. The template itself remains unchanged.

.. plot::
   :caption: Using filters for noise removal.

	import copy
	import numpy as np
	import matplotlib.pyplot as plt
	import matplotlib.colors as colors
	from skimage.util import random_noise

	from tme import Density, Preprocessor
	from tme.matching_data import MatchingData
	from tme.analyzer import MaxScoreOverRotations
	from tme.matching_exhaustive import scan_subsets, MATCHING_EXHAUSTIVE_REGISTER


	def compute_score(
	    target,
	    template,
	    template_mask=None,
	    score="FLC",
	    pad_target_edges: bool = True,
	    pad_fourier: bool = False,
	):
	    if template_mask is None:
	        template_mask = np.ones_like(template)
	    matching_data = MatchingData(
	        target=target.astype(np.float32), template=template.astype(np.float32)
	    )
	    matching_data.template_mask = template_mask
	    matching_data.rotations = np.eye(2).reshape(1, 2, 2)
	    matching_setup, matching_score = MATCHING_EXHAUSTIVE_REGISTER[score]

	    candidates = scan_subsets(
	        matching_data=matching_data,
	        matching_score=matching_score,
	        matching_setup=matching_setup,
	        callback_class=MaxScoreOverRotations,
	        callback_class_args={"score_threshold": -1},
	        pad_target_edges=pad_target_edges,
	        pad_fourier=pad_fourier,
	        job_schedule=(1, 1),
	    )
	    score = candidates[0]
	    score /= score.max()
	    return score


	preprocessor = Preprocessor()
	target = Density.from_file("../../_static/examples/preprocessing_target.png").data
	template_dens = Density.from_file("../../_static/examples/preprocessing_template.png")
	template_dens.data = template_dens.data.astype(np.float32)
	template = template_dens.data

	template_dens.pad(new_shape=target.shape, center=True, padding_value=np.nan)

	fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(10, 10), constrained_layout=True)
	for ax in axs.flat:
	    ax.axis("off")
	np.random.default_rng(42)
	norm = colors.Normalize(vmin=0, vmax=1)

	colormap = copy.copy(plt.cm.gray)
	colormap.set_bad(color="white", alpha=0)
	axs[0, 0].imshow(target, cmap=colormap)
	axs[0, 0].set_title("Target", color="#24a9bb")
	axs[0, 1].imshow(template_dens.data, cmap=colormap)
	axs[0, 1].set_title("Template", color="#24a9bb")
	axs[0, 2].imshow(compute_score(target, template), cmap="viridis", norm=norm)
	axs[0, 2].set_title("Score", color="#24a9bb")

	target_noisy = random_noise(target, mode="gaussian", mean=0, var=0.75)
	target_filter = preprocessor.gaussian_filter(target_noisy, sigma=3)
	axs[1, 0].imshow(target_noisy, cmap="gray")
	axs[1, 0].set_title("Target + Gaussian Noise", color="#24a9bb")
	axs[1, 1].imshow(target_filter, cmap="gray")
	axs[1, 1].set_title("Target Filtered", color="#24a9bb")
	axs[1, 2].imshow(compute_score(target_filter, template), cmap="viridis", norm=norm)
	axs[1, 2].set_title("Score", color="#24a9bb")

	target_noisy = random_noise(target, mode="s&p", amount=0.8)
	target_filter = preprocessor.median_filter(target_noisy, size=9)
	axs[2, 0].imshow(target_noisy, cmap="gray")
	axs[2, 0].set_title("Target + S&P", color="#24a9bb")
	axs[2, 1].imshow(target_filter, cmap="gray")
	axs[2, 1].set_title("Target Filtered", color="#24a9bb")
	score_image = axs[2, 2].imshow(compute_score(target_filter, template), cmap="viridis", norm=norm)
	axs[2, 2].set_title("Score", color="#24a9bb")

	cbar = fig.colorbar(
	    score_image,
	    ax=axs[:, 2],
	    orientation="vertical",
	    location="right",
	    fraction=0.05,
	)

	plt.show()

The noise in the target obfuscates the features the template is designed to detect, consequently leading to a lower score around the position where the template is expected to match accurately. When applying appropriate filters, the scores around the true positive position are elevated, erroneous peaks are suppressed and template matching performance conclusively elevated.

.. note::

   It's pivotal to recognize the complexity of noise in images, often a mix of different types. While Gaussian filters are widely used and generally effective, exploring a variety of filters tailored to specific noise types can yield optimal results in noise reduction and template matching accuracy.

   	This section was inspired by an `OpenCV tutorial <https://docs.opencv.org/4.9.0/d4/dc6/tutorial_py_template_matching.html>`_, which discusses template matching using different variations of the normalized cross-correlation coefficient.


Component Emphasis
------------------

Building on our understanding of noise removal in preprocessing, we now shift our focus to emphasizing specific components within an image. Template matching makes heavy use of the Fourier transform, which maps an image from real space into a spectrum of frequencies, each with a corresponding magnitude and phase, rather than intensities. Lower frequencies correspond to broad features, while higher frequencies represent finer detail and noise. By selectively enhancing or suppressing specific frequency components, we can emphasize desired features or reduce undesirable noise in an image.


Frequency Filter
^^^^^^^^^^^^^^^^

Low-pass, high-pass and band-pass filters serve as prototypical modulators of an objects's frequency components. Low-pass filters allow frequency up to a certain threshold to pass through, effectively smoothing the image by removing finer details and high-frequency noise. High-pass filters do the opposite by allowing only the higher frequencies to pass, this enhancing fine details. Band-pass filters represent a combination of low and high-pass, allowing a specific range of frequencies to pass.


.. plot::
   :caption: Application of Frequency Filters.

   from skimage.feature import match_template

   from tme import Density, Preprocessor
   from tme.preprocessing import BandPassFilter

   target = Density.from_file("../../_static/examples/preprocessing_target.png").data
   target_ft = np.fft.fftshift(np.fft.fft2(target))
   template = Density.from_file("../../_static/examples/preprocessing_template.png").data

   preprocessor = Preprocessor()
   fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 5), constrained_layout=True)
   for ax in axs.flat:
      ax.axis("off")
   lowpass = BandPassFilter(
      lowpass = 5,
      highpass = None,
      sampling_rate = 1,
   )(shape = target.shape, shape_is_real_fourier = False, return_real_fourier=False)["data"]
   lowpass = np.fft.ifftshift(lowpass)
   target_lowpass = np.fft.ifft2(np.fft.ifftshift(lowpass * target_ft)).real

   highpass = BandPassFilter(
      lowpass = None,
      highpass = 4,
      sampling_rate = 1,
   )(shape = target.shape, shape_is_real_fourier = False, return_real_fourier=False)["data"]
   highpass = np.fft.fftshift(highpass)
   target_highpass = np.fft.ifft2(np.fft.ifftshift(highpass * target_ft)).real

   bandpass = BandPassFilter(
      lowpass = 5,
      highpass = 10,
      sampling_rate = 1
   )(shape = target.shape, shape_is_real_fourier = False, return_real_fourier=False)["data"]
   bandpass = np.fft.fftshift(bandpass)
   target_bandpass = np.fft.ifft2(np.fft.ifftshift(bandpass * target_ft)).real

   score = match_template(target_lowpass, template)
   axs[0, 0].imshow(target_lowpass, cmap='gray')
   axs[0, 0].set_title('Low-pass Filtered', color = '#24a9bb')
   axs[1, 0].imshow(score / score.max())
   axs[1, 0].set_title('Score (Low-pass)', color = '#24a9bb')

   score = match_template(target_highpass, template)
   axs[0, 1].imshow(target_highpass, cmap='gray')
   axs[0, 1].set_title('High-pass Filtered', color = '#24a9bb')
   axs[1, 1].imshow(score / score.max())
   axs[1, 1].set_title('Score (High-pass)', color = '#24a9bb')

   score = match_template(target_bandpass, template)
   axs[0, 2].imshow(target_bandpass, cmap='gray')
   axs[0, 2].set_title('Band-pass Filtered', color = '#24a9bb')
   axs[1, 2].imshow(score / score.max())
   axs[1, 2].set_title('Score (Band-pass)', color = '#24a9bb')

   plt.show()

The high-pass and band-pass filtered images exhibit sharp pronounced peaks and an overall reduction in the background scores, while the low-pass filter yields a wide peak. The emphasis on low-frequency information creates a situation in which overall template shape and size rather than structural detail drive the detection [1]_.


Spectral Whitening
^^^^^^^^^^^^^^^^^^

Spectral whitening normalizes each frequency by dividing the amplitude of each frequency by its magnitude. This can bring out subtle features that might be overshadowed otherwise. Spectral Whitening is particulary useful when analyzing large heterogeneous datasets, because the assumptions made are fairly weak. Although not specific to cryogenic electron microscopy, spectral whitening is a fairly popular [2]_ [3]_ approach and graphically explained `here <https://github.com/ZauggGroup/DeePiCt/blob/main/spectrum_filter/tomo-matcher.svg>`_.


.. plot::
   :caption: Application of Spectral Whitening.

   import copy
   from skimage.feature import match_template

   from tme import Density
   from tme.preprocessing import LinearWhiteningFilter

   target = Density.from_file("../../_static/examples/preprocessing_target.png").data
   template = Density.from_file(
      "../../_static/examples/preprocessing_template.png"
   ).data

   whitening_filter = LinearWhiteningFilter()
   target_filter = whitening_filter(data = target)["data"]
   template_filter = whitening_filter(data = template)["data"]

   target_filtered = np.fft.irfftn(np.fft.rfftn(target) * target_filter)
   template_filtered = np.fft.irfftn(np.fft.rfftn(template) * template_filter)

   fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(10, 5), constrained_layout=True)
   for ax in axs.flat:
      ax.axis("off")

   score = match_template(target_filtered, template_filtered, pad_input = True)
   axs[0].imshow(target_filtered, cmap = "gray")
   axs[0].set_title('Whitened', color = '#24a9bb')
   axs[1].imshow(score / score.max())
   axs[1].set_title('Score (Spectral Whitening)', color = '#24a9bb')

   plt.show()

Spectral Whitening leads to a more similar frequency composition of the template and target, thus reducing bias towards certain frequency ranges. This in turn results in a sharp peak in template matching score around the true positive location and a reduction in background scores compared to the unfiltered template matching scores.


CTF
^^^

The contrast transfer (`CTF <https://guide.cryosparc.com/processing-data/all-job-types-in-cryosparc/ctf-estimation>`_) function is a mathematical description of the modulation incurred to a specimen when viewed through a microscope. In cryogenic electron microscopy we are often concerned with correction for the CTF, but it can also be used to simulate how a given template would be observed through a microscope. Accounding for the CTF thus improves template matching by increasing the similarity between the template and template instances in the target.

Shown below is how we can use the CTF to inspect an object at different defocus values. However, in practice there are more parameters to consider as we will discuss in a following tutorial.

.. plot::
   	:caption: Application of CTF.

	import copy
	import numpy as np
	import matplotlib.pyplot as plt

	from skimage.feature import match_template

	from tme import Density
	from tme.preprocessing.tilt_series import CTF

	target = Density.from_file("../../_static/examples/preprocessing_target.png")

	ctf = CTF(
	    shape=target.shape,
	    angles=[0],
	    sampling_rate=target.sampling_rate[0],
	    acceleration_voltage=200 * 1e3,
	    defocus_x=[1000],
	    spherical_aberration=2.7e7,
	    amplitude_contrast=0.08,
	    flip_phase=False,
	    return_real_fourier=True,
	)


	fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 5), constrained_layout=True)
	for ax in axs.flat:
	    ax.axis("off")

	ctf_mask = ctf()["data"]
	target_filtered = np.fft.irfftn(np.fft.rfftn(target.data) * ctf_mask)
	axs[0].imshow(target_filtered, cmap="gray")
	axs[0].set_title("Defocus 1000", color="#24a9bb")

	ctf.defocus_x[0] = 2500
	ctf_mask = ctf()["data"]
	target_filtered = np.fft.irfftn(np.fft.rfftn(target.data) * ctf_mask)
	axs[1].imshow(target_filtered, cmap="gray")
	axs[1].set_title("Defocus 2500", color="#24a9bb")

	ctf.defocus_x[0] = 5000
	ctf_mask = ctf()["data"]
	target_filtered = np.fft.irfftn(np.fft.rfftn(target.data) * ctf_mask)
	axs[2].imshow(target_filtered, cmap="gray")
	axs[2].set_title("Defocus 5000", color="#24a9bb")
	plt.show()


Wedge Masks
^^^^^^^^^^^

Cryogenic electron tomography is based on the reconstruction of volumes from a set of 2D images obtained at different tilt angles. The number as well as the range of angles that can be sample is finite, due to sample and stage limitations. Therefore, the Fourier space of tomograms is hallmarked by regions of missing information, which is commonly summarized as missing-wedge. Through wedge masks, we can give the same characteristic to the template, thus excluding information-free regions from the matching procedure.

Broadly speaking, |project| distinguishes between continuous, discrete and weighted wedge masks. The later introduces angular-dependent frequency specific weighting and can be done in a variety of ways.

.. plot::
   	:caption: Application of Wedge Masks.

	from tme import Density
	from tme.preprocessing.tilt_series import WedgeReconstructed

	wedge = WedgeReconstructed(
	    angles = [60,60],
	    opening_axis = 0,
	    tilt_axis = 1,
	    create_continuous_wedge = True,
	    weight_wedge = False,
	    reconstruction_filter = "cosine"
	)

	fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 5), constrained_layout=True)
	for ax in axs.flat:
	    ax.axis("off")

	mask = wedge(shape = (100,100), return_real_fourier = False)["data"]
	axs[0].imshow(np.fft.fftshift(mask), cmap = "gray")
	axs[0].set_title('Continuous Wedge', color = '#24a9bb')

	wedge.create_continuous_wedge = False
	wedge.angles = np.linspace(-60, 60, 20)
	mask = wedge(shape = (100,100), return_real_fourier = False)["data"]
	axs[1].imshow(np.fft.fftshift(mask), cmap = "gray")
	axs[1].set_title('Discrete Wedge', color = '#24a9bb')

	wedge.weight_wedge = True
	mask = wedge(shape = (100,100), return_real_fourier = False)["data"]
	axs[2].imshow(np.fft.fftshift(mask), cmap = "gray")
	axs[2].set_title('Weighted Discrete Wedge', color = '#24a9bb')

	plt.show()


References
----------

.. [1] Maurer, V. J.; Siggel, M.; Kosinski, J. What shapes template-matching performance in cryogenic electron tomography in situ?. Acta Crys D 2024
.. [2] de Teresa-Trueba, I.; Goetz, S. K.; Mattausch, A.; Stojanovska, F.; Zimmerli, C. E.; Toro-Nahuelpan, M.; Cheng, D. W. C.; Tollervey, F.; Pape, C.; Beck, M.; Diz-Munoz, A.; Kreshuk, A.; Mahamid, J.; Zaugg, J. B. Convolutional networks for supervised mining of molecular patterns within cellular context. Nat. Methods 2023, 20, 284–294.
.. [3] Grant, T.; Rohou, A.; Grigorieff, N. cisTEM, user-friendly software for single-particle image processing. eLife 2018, 7, e35383.