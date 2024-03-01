.. include:: ../substitutions.rst

=============
Preprocessing
=============

TL;DR
-----

The ``preprocessor_gui.py`` provides a graphical user interface for common preprocessing operations such as filtering or the creation of masks for template matching. After installation as outlined in the :ref:`gui-installation`, you can launch it via:

.. code-block:: bash

    preprocessor_gui.py

Users that aim to perform these operations programmatically can do so via :py:class:`tme.density.Density`, :py:class:`tme.preprocessor.Preprocessor`, :py:meth:`tme.matching_utils.create_mask`. For further details refer to :ref:`filter-application`.

Aim
---

Cross-correlation based template matching is only as good as the data allows it to be. Often times, cross-correlation will result in erroneous matches, although the correct result is evident by eye. Notably, this is not because cross-correlation is a particularly bad similarity metric, but rather owed to the excellent template matching capacity and exponential escalation of the human mind.

Nevertheless, we can preprocess our data to guide cross-correlation based template matching in the right direction using techniques outliens in the following sections. With preprocessing, we aim to improve template matching accuracy, by increasing the cross-correlation score given to true positive instances of the template, while decreasing the cross-correlation score of false positives.


Background
----------

In order to understand which factors to focus on during preprocessing, we have to take a schematic look at the computation of the normalized cross-correlation

.. math::
   :name: eq:normcc

   score = \frac{(x - \mu)}{\sigma}.

- :math:`x` represents the non-normalized cross-correlation between the target and the template.

- :math:`\mu` represents the non-normalized cross-correlation between the target and a mask.

- :math:`\sigma` represents the variance in non-normalized cross-correlation between the target and a mask.

- :math:`score` represents the normalized cross-correlation which is clipped between zero and one, with one being a perfect match.


Based on this schematic assessment, we can influence the normalized cross-correlation score through two principal approaches:

1. :ref:`Apply filters<preprocess-filtering>` to improve the similarity between the template and template instances in the target.

2. :ref:`Design a mask<mask-design>` that provides a faithful approximation of the cross-correlation background given the template.

These preprocessing approaches will be discussed in the following sections. For a mathematically more rigorous treatment of the normalized cross-correlation we refer the reader to [1]_.


.. _preprocess-filtering:

Filtering Use Cases
-------------------

The goal of preprocessing is to improve the similarity between the template and template instances in the target. This should result in a better detection and in turn a higher precision of the template matching procedure. In the following we focus on two particular applications of filters:

1. to remove noise from the data.

2. to emphasize particular components of the data.

Let us better understand these applications through a practical template matching example using the target (left) and template (right) displayed below. Typically template matching also involves evaluating the similarity of different rotations of the template, which we omit because it is not necessary in order to understand the preprocessing concepts.

.. plot::
   :caption: Template matching data used in this guide.

   import copy

   import numpy as np
   import matplotlib.pyplot as plt

   from tme import Density

   target = Density.from_file("../_static/examples/preprocessing_target.png").data
   target = target.astype(np.float32)
   template = Density.from_file("../_static/examples/preprocessing_template.png")
   template.data = template.data.astype(np.float32)
   template.pad(new_shape = target.shape, center = True, padding_value = np.nan)
   template = template.data

   fig, axs = plt.subplots(
      nrows = 1,
      ncols = 2,
      sharex=True,
      sharey=True,
      figsize = (12,5)
   )

   colormap = copy.copy(plt.cm.gray)
   colormap.set_bad(color='white', alpha=0)

   axs[1].imshow(template, cmap=colormap)
   axs[1].set_title('Template', color = '#0a7d91')
   axs[1].axis('off')

   axs[0].imshow(target, cmap=colormap)
   axs[0].set_title('Target', color='#0a7d91')
   axs[0].axis('off')

   plt.tight_layout()
   plt.show()


Noise Removal
^^^^^^^^^^^^^

Real-world data often contain noise, arising from varied sources like sensor imperfections, environmental conditions, or the intricacies of the image acquisition process. This noise can obscure or distort the features of the image that are essential for accurate template matching. Filters are instrumental in mitigating the impact of noise. Filters allow us to reduce the amount of noise. In the following, we explore the impact of two common types of noise: Gaussian and salt-and-pepper (S&P). We then compare the template matching scores with and without the application of appropriate filters before template matching. The template itself remains unchanged.

.. plot::
   :caption: Assessment of different filters for noise removal prior to template matching.

   import numpy as np
   import matplotlib.pyplot as plt
   import matplotlib.colors as colors
   from skimage.util import random_noise
   from skimage.feature import match_template

   from tme import Density, Preprocessor
   from tme.matching_data import MatchingData
   from tme.analyzer import MaxScoreOverRotations
   from tme.matching_exhaustive import scan_subsets, MATCHING_EXHAUSTIVE_REGISTER

   def compute_score(
      target,
      template,
      template_mask = None,
      score = "FLC",
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
      return candidates[0]

   preprocessor = Preprocessor()
   target = Density.from_file("../_static/examples/preprocessing_target.png").data
   template = Density.from_file("../_static/examples/preprocessing_template.png").data

   fig, axs = plt.subplots(
      nrows=5,
      ncols=2,
      sharex=True,
      sharey=True,
      figsize=(12, 25),
      constrained_layout=True
   )

   np.random.default_rng(42)
   norm = colors.Normalize(vmin=0, vmax=1)

   template_mask = np.ones_like(template)
   axs[0, 0].imshow(target, cmap = "gray")
   axs[0, 0].set_title('Target', color = '#0a7d91')
   axs[0, 1].imshow(compute_score(target, template))
   axs[0, 1].set_title('Template Matching Score', color = '#0a7d91')

   target_noisy = random_noise(target, mode="gaussian", mean = 0, var = .75)
   axs[1, 0].imshow(target_noisy, cmap = "gray")
   axs[1, 0].set_title('Target + Gaussian Noise', color = '#0a7d91')
   axs[1, 1].imshow(compute_score(target_noisy, template),
      cmap='viridis', norm=norm)
   axs[1, 1].set_title('Template Matching Score', color = '#0a7d91')

   target_filter = preprocessor.gaussian_filter(target_noisy, sigma = 1)
   axs[2, 0].imshow(target_filter, cmap = "gray")
   axs[2, 0].set_title('Target + Gaussian Noise + Gaussian Filter', color = '#0a7d91')
   axs[2, 1].imshow(compute_score(target_filter, template),
      cmap='viridis', norm=norm)
   axs[2, 1].set_title('Template Matching Score', color = '#0a7d91')

   target_noisy = random_noise(target, mode="s&p", amount = .7)
   axs[3, 0].imshow(target_noisy, cmap = "gray")
   axs[3, 0].set_title('Target + S&P Noise', color = '#0a7d91')
   axs[3, 1].imshow(compute_score(target_noisy, template),
      cmap='viridis', norm=norm)
   axs[3, 1].set_title('Template Matching Score', color = '#0a7d91')

   target_filter = preprocessor.median_filter(target_noisy, size = 7)
   axs[4, 0].imshow(target_filter, cmap = "gray")
   axs[4, 0].set_title('Target + S&P Noise + Median Filter', color = '#0a7d91')
   score_image=axs[4, 1].imshow(compute_score(target_filter, template),
      cmap='viridis', norm=norm)
   axs[4, 1].set_title('Template Matching Score', color = '#0a7d91')

   cbar = fig.colorbar(
      score_image,
      ax = axs[:, 1],
      orientation='vertical',
      location = "right",
      fraction=0.05,
   )

   plt.show()

.. note::

   The normalized cross-correlation score of pixels close to boundaries are often elevated. This is a common artifact since these areas do not allow for cross-correlation computation using the entire template, they can be masked. Alternatively ``postprocess.py`` (see :doc:`postprocessing`) has a ``min_boundary_distance`` flag, that disregrads peaks in this area.


When the target image is infused with noise, the template matching score around the true positive position (where the template is expected to match accurately) is notably lower. This decrease in the matching score is indicative of the noise interfering with the ability to find correct matches in the target image. The presence of noise essentially masks or distorts the features that the template is designed to detect, leading to a reduced confidence in identifying the correct location. Furthermore, the presence of noise introduces additional bias, such as erroneous peaks in the template matching score. These peaks are not present in the original, noise-free image, and potentially result in the detection of false positives. In other words, the template might incorrectly identify parts of the noisy image as matches, even though these areas do not actually resemble the template.

When applying appropriate filters to the noisy data, the template matching score around the true positive position is elevated and erroneous peaks are suppressed, reducing the likelihood of false positives. The filtering process helps in smoothing out the noise, thereby reducing its impact.

Conclusively, effective noise reduction through filtering can significantly enhance template matching performance. However, it's pivotal to recognize the complexity of noise in images, often a mix of different types. While Gaussian filters are widely used and generally effective, exploring a variety of filters tailored to specific noise types can yield optimal results in noise reduction and template matching accuracy.

.. note::

   This section was inspired by an `OpenCV tutorial <https://docs.opencv.org/4.9.0/d4/dc6/tutorial_py_template_matching.html>`_, which discusses template matching using different variations of the normalized cross-correlation coefficient.


Component Emphasis
^^^^^^^^^^^^^^^^^^

Building on our understanding of noise removal in preprocessing, we now shift our focus to emphasizing specific components within an image. After addressing noise with filters like Gaussian and median filters, the next step often involves highlighting or isolating particular features or structures in the data through frequency space operations. The Fourier transform is integral to understanding the algorithmic underpinning of template matching, which is futher discussed :ref:`here <match-template-background>`.

The Fourier transform translates an image from real space into a spectrum of frequencies, each with a corresponding magnitude and phase, rather than intensities. The real space image (left) is decomposed into its constituent frequencies (right). The center of the Fourier-transformed image represents the zero frequency component, often referred to as the DC component, which signifies the average intensity of the image. As we move outward from the center, we encounter higher frequencies. Notably, all points on a given ellipse correspond to the same frequency, but differ in orientation. For input images with equal dimensions, i.e. square images, points of identical frequency would lie on a circle.

The distance from the center is directly proportional to the frequency magnitude: closer points represent lower frequencies (broad features), while farther points represent higher frequencies (fine details and noise). By selectively enhancing or suppressing specific frequency components, we can emphasize desired features or reduce undesirable noise in an image.

.. plot::
   :caption: Introduction to the Fourier transform.

   import numpy as np
   import matplotlib.pyplot as plt
   import matplotlib.patches as mpatches

   from tme import Density

   target = Density.from_file("../_static/examples/preprocessing_target.png").data
   target_fft = np.fft.fftshift(np.fft.fft2(target))

   fig, axs = plt.subplots(1, 2, figsize=(12, 5))

   axs[0].imshow(target, cmap='gray')
   axs[0].set_title('Target', color = '#0a7d91')

   axs[1].imshow(np.log(np.abs(target_fft)), cmap='gray')
   axs[1].set_title('Fourier Transform Magnitude', color = '#0a7d91')

   radius = 50
   center = np.divide(target_fft.shape, 2).astype(int)[::-1]

   aspect_ratio = target_fft.shape[1] / target_fft.shape[0]

   for i in range(1,4):
      radius_y = i * 50  # Radius in the x direction
      radius_x = radius_y * aspect_ratio  # Adjusted radius in the y direction
      ellipse = mpatches.Ellipse(center, 2*radius_x, 2*radius_y, color='red', fill=False)
      axs[1].add_patch(ellipse)


   arrow_stop = center.copy()
   arrow_stop[0] *= 1.75

   arr = mpatches.FancyArrowPatch(
      center, arrow_stop,
      arrowstyle='->,head_width=.15',
      mutation_scale=20
   )
   axs[1].add_patch(arr)
   axs[1].annotate(
      "Frequency ↑", (.5, .5), xycoords=arr, ha='center', va='bottom'
   )

   plt.tight_layout()
   plt.show()


In frequency domain image processing, filters like low-pass, high-pass, and bandpass serve as prototypical modulators, each uniquely altering the image's frequency components.

- Low-pass filters allow only the lower frequencies to pass through, effectively smoothing the image by removing finer details and high-frequency noise. This results in a focus on broader, general features of the image.

- High-pass filters do the opposite by allowing only the higher frequencies to pass. They enhance fine details and edges, often at the expense of the general layout or broader features.

- Bandpass filters offer a more selective approach, allowing a specific range of frequencies to pass. This specificity enables the isolation of features of particular sizes or the smoothing out of noise within a certain frequency band.

By multiplying the frequency components of an image using these filters, followed by an inverse Fourier Transform, we can precisely control how features of various scales are emphasized or suppressed. This process effectively translates the modifications made in the frequency domain back to the spatial domain.

.. plot::
   :caption: Emphasizing image components using common Fourier filters.

   import numpy as np
   import matplotlib.pyplot as plt
   from tme import Density, Preprocessor

   target = Density.from_file("../_static/examples/preprocessing_target.png").data
   target_ft = np.fft.fftshift(np.fft.fft2(target))

   preprocessor = Preprocessor()
   fig, axs = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)

   lowpass = np.fft.fftshift(preprocessor.bandpass_mask(
      shape = target_ft.shape,
      minimum_frequency = 0,
      maximum_frequency = .1,
      omit_negative_frequencies = False,
      gaussian_sigma = 2
   ))
   target_lowpass = np.fft.ifft2(np.fft.ifftshift(lowpass * target_ft)).real

   highpass = np.fft.fftshift(preprocessor.bandpass_mask(
      shape = target_ft.shape,
      minimum_frequency = .2,
      maximum_frequency = .75,
      omit_negative_frequencies = False,
      gaussian_sigma = 2
   ))
   target_highpass = np.fft.ifft2(np.fft.ifftshift(highpass * target_ft)).real

   bandpass = np.fft.fftshift(preprocessor.bandpass_mask(
      shape = target_ft.shape,
      minimum_frequency = .1,
      maximum_frequency = .4,
      omit_negative_frequencies = False,
      gaussian_sigma = 2
   ))
   target_bandpass = np.fft.ifft2(np.fft.ifftshift(bandpass * target_ft)).real

   axs[0, 0].imshow(lowpass)
   axs[0, 0].set_title('Low-pass Filter', color = '#0a7d91')
   axs[0, 1].imshow(target_lowpass, cmap='gray')
   axs[0, 1].set_title('Low-pass Filtered', color = '#0a7d91')

   axs[1, 0].imshow(highpass)
   axs[1, 0].set_title('High-pass Filter', color = '#0a7d91')
   axs[1, 1].imshow(target_highpass, cmap='gray')
   axs[1, 1].set_title('High-pass Filtered', color = '#0a7d91')

   axs[2, 0].imshow(bandpass)
   axs[2, 0].set_title('Band-pass Filter', color = '#0a7d91')
   axs[2, 1].imshow(target_bandpass, cmap='gray')
   axs[2, 1].set_title('Band-pass Filtered', color = '#0a7d91')

   plt.show()

Naturally, the emphasis on particular components has profound effects on the template matching score. The high-pass and band-pass filtered images exhibit sharp pronounced peaks and an overall reduction in the background scores. This is because the template matching score is dominated by low-frequency information, thus creating a situation in which overall template shape and size rather than structural detail drive the detection. For a more complete description and assessment of this phenomenon in the context of cryogenic electron tomography, we refer the reader to [2]_.

.. plot::
   :caption: Influence of common Fourier Filters on template matching scores.

   import numpy as np
   import matplotlib.pyplot as plt
   from skimage.feature import match_template

   from tme import Density, Preprocessor

   target = Density.from_file("../_static/examples/preprocessing_target.png").data
   template = Density.from_file("../_static/examples/preprocessing_template.png").data

   target_ft = np.fft.fftshift(np.fft.fft2(target))

   preprocessor = Preprocessor()
   fig, axs = plt.subplots(3, 1, figsize=(12, 15), constrained_layout=True)

   lowpass = np.fft.fftshift(preprocessor.bandpass_mask(
      shape = target_ft.shape,
      minimum_frequency = 0,
      maximum_frequency = .1,
      omit_negative_frequencies = False,
      gaussian_sigma = 2
   ))
   target_lowpass = np.fft.ifft2(np.fft.ifftshift(lowpass * target_ft)).real

   highpass = np.fft.fftshift(preprocessor.bandpass_mask(
      shape = target_ft.shape,
      minimum_frequency = .2,
      maximum_frequency = .75,
      omit_negative_frequencies = False,
      gaussian_sigma = 2
   ))
   target_highpass = np.fft.ifft2(np.fft.ifftshift(highpass * target_ft)).real

   bandpass = np.fft.fftshift(preprocessor.bandpass_mask(
      shape = target_ft.shape,
      minimum_frequency = .1,
      maximum_frequency = .4,
      omit_negative_frequencies = False,
      gaussian_sigma = 2
   ))
   target_bandpass = np.fft.ifft2(np.fft.ifftshift(bandpass * target_ft)).real

   axs[0].imshow(match_template(target_lowpass, template))
   axs[0].set_title('Template Matching Score (Low-pass)', color = '#0a7d91')

   axs[1].imshow(match_template(target_highpass, template))
   axs[1].set_title('Template Matching Score (High-pass)', color = '#0a7d91')

   axs[2].imshow(match_template(target_bandpass, template))
   axs[2].set_title('Template Matching Score (Band-pass)', color = '#0a7d91')

   plt.show()

.. note::

   Although the application of a filter is commutative in Fourier space, this does not hold for the mask that normalizes the cross-correlation coefficient. Therefore, if the goal is to have the filtering effect reflected in the normalization procedure, the filter needs to be applied to the target.


The possibilities of defining filters in Fourier space is quasi-unlimited and can include far more intricate or periodic filters, such as tilt weighting for tomography or contrast transfer functions (`CTF <https://guide.cryosparc.com/processing-data/all-job-types-in-cryosparc/ctf-estimation>`_).

To showcase another possibility of placing emphasis on particular image components, we take a look at spectral whitening, which is particulary useful when analyzing large heterogeneous datasets, because the assumptions made are fairly weak. Although neither new nor specific to cryogenic electron microscopy, spectral whitening is a fairly popular [3]_ [4]_ approach and graphically explained `here <https://github.com/ZauggGroup/DeePiCt/blob/main/spectrum_filter/tomo-matcher.svg>`_. Briefly, spectral whitening normalizes each frequency by dividing the amplitude of each frequency by its magnitude. This can bring out subtle features that might be overshadowed otherwise [2]_. In the context of template matching, we can enforce a more similar frequency composition of template and target, thus reducing bias towards certain frequency ranges. This in turn results in a sharp peak in template matching score around the true positive location and a reduction in background scores compared to the unfiltered template matching scores.

.. plot::
   :caption: Influence of spectral whitening on template matching scores.

   import copy

   import numpy as np
   import matplotlib.pyplot as plt
   from skimage.feature import match_template

   from tme import Density
   from tme.preprocessor import LinearWhiteningFilter

   target = Density.from_file("../_static/examples/preprocessing_target.png").data
   template = Density.from_file(
      "../_static/examples/preprocessing_template.png"
   ).data

   whitening_filter = LinearWhiteningFilter()
   target_filtered, bins, averages = whitening_filter.filter(target)
   template_filtered, bins, averages = whitening_filter.filter(template)

   fig, axs = plt.subplots(nrows = 1, ncols = 1)

   axs.imshow(match_template(
      target_filtered, template_filtered, pad_input = True)
   )
   axs.set_title('Template Matching Score (Spectral Whitening)', color = '#0a7d91')

   plt.tight_layout()
   plt.show()

.. _mask-design:

Mask Design
-----------

As outlined in :ref:`equation 1 <eq:normcc>`, the mask is pivotal for the normalization of the cross-correlation coefficient. The composition of an optimal mask is problem-specific, but we recommend the following when designing masks:

1. The mask should focus on conserved key features.

2. Consider the context in which the mask will be applied. Particularly for *in situ* images, crowdedness can be problematic if the mask is excessively large.

3. Sharp transitions or edge effects should be avoided by smoothing or sufficiently sized boxes.


To build intuition towards mask design, let us assess the effect of a range of binary masks outlined below. By default, the entire area under the template is considered as mask. While this results in a properly normalized cross-correlation coefficient, it comes with downfalls: 1.) The cross-correlation peak is wide, which is disadvantageous in crowded settings as it may lead to the obfuscation of peaks. 2.) The background score is high, which can yield false positive peaks. When focusing on defining features of the template using an ellipsoidal mask, i.e. eyes, nose and mouth, we observe a sharpening of the peak and a decrease in background score. However, if we make the mask too narrow, as illustrated by the last example, the mask no longer provides a faithful approximation of the cross-correlation background and as thus the template matching scores become uninformative.

.. plot::
   :caption: Influence of mask design on template matching scores.

   import copy

   import numpy as np
   import matplotlib.pyplot as plt

   from tme import Density
   from tme.matching_utils import create_mask
   from tme.matching_data import MatchingData
   from tme.analyzer import MaxScoreOverRotations
   from tme.matching_exhaustive import scan_subsets, MATCHING_EXHAUSTIVE_REGISTER


   def compute_score(target, template, template_mask, score):
       matching_data = MatchingData(
           target=target.astype(np.float32),
           template=template.astype(np.float32)
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
           pad_target_edges=False,
           pad_fourier=False,
           job_schedule=(1,1),
       )
       return candidates[0]


   if __name__ == "__main__":

       matching_score = "FLCSphericalMask"

       target = Density.from_file("../_static/examples/preprocessing_target.png").data
       template = Density.from_file("../_static/examples/preprocessing_template.png").data
       target = target.astype(np.float32)
       template = template.astype(np.float32)

       fig, axs = plt.subplots(
           nrows=3, ncols=2, sharex=False, sharey=False, figsize=(12, 15)
       )
       colormap = copy.copy(plt.cm.gray)
       colormap.set_bad(color='white', alpha=0)

       template_mask = np.ones_like(template)
       score = compute_score(
           target=target,
           template=template,
           template_mask=template_mask,
           score=matching_score,
       )
       masked_template  = template.copy()
       masked_template[template_mask == 0] = np.nan
       axs[0, 0].imshow(masked_template, cmap=colormap)
       axs[0, 1].imshow(score)
       axs[0, 0].set_title("Template + Default Mask", color="#0a7d91")
       axs[0, 1].set_title("Template Matching Score", color="#0a7d91")

       mask_center = np.add(
           np.divide(template.shape, 2).astype(int), np.mod(template.shape, 2)
       )
       template_mask = create_mask(
           mask_type="ellipse", radius=(20, 10), shape=template.shape, center=mask_center
       )
       score = compute_score(
           target=target,
           template=template,
           template_mask=template_mask,
           score=matching_score,
       )
       masked_template  = template.copy()
       masked_template[template_mask == 0] = np.nan
       axs[1, 0].imshow(masked_template, cmap=colormap)
       axs[1, 1].imshow(score)
       axs[1, 0].set_title("Template + Ellipsoidal Mask", color="#0a7d91")
       axs[1, 1].set_title("Template Matching Score", color="#0a7d91")

       template_mask = create_mask(
           mask_type="ellipse", radius=(5, 5), shape=template.shape, center=mask_center
       )
       score = compute_score(
           target=target,
           template=template,
           template_mask=template_mask,
           score=matching_score,
       )
       masked_template  = template.copy()
       masked_template[template_mask == 0] = np.nan
       axs[2, 0].imshow(masked_template, cmap=colormap)
       axs[2, 1].imshow(score)
       axs[2, 0].set_title("Template + Spherical Mask", color="#0a7d91")
       axs[2, 1].set_title("Template Matching Score", color="#0a7d91")

       plt.tight_layout()
       plt.show()

The applied ellipsoidal mask is a suitable choice in this case, because the mask recapitulates the defining features of a face well. Therefore, we also avoid sharp transition in template intensity that could adversely effect the template matching score computation. In the following we look at the effect of smoothing a suboptimal mask, that includes strong changes in intensity. For smoothing the mask, we apply a Gaussian filter that rounds the edges and provides a smoother intensity transition. Albeit difficult to see in this representation, smoothing the mask results in 15% higher peaks, compared to their individual backgrounds. Nevertheless, the benefit of smoothing masks is less obvious and has to be evaluated on a per-problem basis.

.. plot::
   :caption: Influence of mask smoothing on template matching scores.

   import copy

   import numpy as np
   import matplotlib.pyplot as plt

   from tme import Density
   from tme.matching_utils import create_mask
   from tme.matching_data import MatchingData
   from tme.analyzer import MaxScoreOverRotations
   from tme.matching_exhaustive import scan_subsets, MATCHING_EXHAUSTIVE_REGISTER

   def compute_score(target, template, template_mask, score):
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
         pad_target_edges=False,
         pad_fourier=False,
         job_schedule=(1,1),
      )
      return candidates[0]


   if __name__ == "__main__":

      matching_score = "FLCSphericalMask"

      target = Density.from_file("../_static/examples/preprocessing_target.png").data
      template = Density.from_file(
        "../_static/examples/preprocessing_template.png"
      ).data
      target = target.astype(np.float32)
      template = template.astype(np.float32)

      fig, axs = plt.subplots(
        nrows=2, ncols=2, sharex=False, sharey=False, figsize=(12, 10)
      )
      colormap = copy.copy(plt.cm.gray)
      colormap.set_bad(color="white", alpha=0)

      mask_center = np.add(
        np.divide(template.shape, 2).astype(int), np.mod(template.shape, 2)
      )
      mask_center[0] = mask_center[0] - 5
      mask_center[1] = mask_center[1] - 5
      template_mask = create_mask(
         mask_type="box",
         height=(10, 18),
         shape=template.shape,
         center=mask_center,
      )
      score = compute_score(
         target=target,
         template=template,
         template_mask=template_mask,
         score=matching_score,
      )

      masked_template = template.copy()
      masked_template[template_mask == 0] = np.nan
      axs[0, 0].imshow(masked_template, cmap=colormap)
      axs[0, 1].imshow(score)
      axs[0, 0].set_title("Template + Box Mask", color="#0a7d91")
      axs[0, 1].set_title("Template Matching Score", color="#0a7d91")

      template_mask = create_mask(
         mask_type="box",
         height=(10, 18),
         shape=template.shape,
         center=mask_center,
         sigma_decay=2,
      )
      score = compute_score(
         target=target,
         template=template,
         template_mask=template_mask,
         score=matching_score,
      )
      masked_template = template.copy()
      masked_template[template_mask == 0] = np.nan
      axs[1, 0].imshow(masked_template, cmap=colormap)
      axs[1, 1].imshow(score)
      axs[1, 0].set_title("Template + Smoothed Box Mask", color="#0a7d91")
      axs[1, 1].set_title("Template Matching Score", color="#0a7d91")

      plt.tight_layout()
      plt.show()


Generally, the :ref:`filtering procedure <preprocess-filtering>` has greater impact on template matching performance than the choice of mask. In practice, it suffices to use a geometric object that recapitulates the size and shape of the template [2]_. Examples from the literature include ellipsoids or spheres for ribosomes [3]_, cylinders for nucleasomes [6]_ or proteasomes [7]_ and boxes for membranes [8]_.

Noteably, the choice of mask does have an impact on the runtime performance of template matching. Rotation symmetric masks, or masks that encompass all possible rotations of the template do not need to be rotated during template matching, thus reducing the runtime by up to a factor of three.

.. note::

   A more complete treatment of the mathematical implications of using different masks is provided in [5]_, together with `explanatory slides <https://pdfs.semanticscholar.org/17e5/419d9eb239b91b46fde52538e6c13b33909a.pdf>`_. Alternatively, `this OpenCV tutorial <https://docs.opencv.org/3.4/de/da9/tutorial_template_matching.html>`_ provides further examples on the effect of masking.


.. _filter-application:

Practical Example
-----------------

The following elaborates how to perform preprocessing using the graphical user interface (GUI) and the application programming interface (API). While the GUI provides access to the most commonly used operations, the API gives access to all components of |project| for developers.

.. tab-set::

   .. tab-item:: GUI

      The following command will launch a `napari` viewer embedded with custom widgets for preprocessing. For a detailed description of all GUI elements please have a look at the `napari documentation <https://napari.org/stable/tutorials/fundamentals/quick_start.html>`_. Note that using the GUI requires following the installation procedure outlined in the :ref:`gui-installation`.

      .. code-block:: bash

         preprocessor_gui.py

      The launched `napari` viewer instance will appear similar to the figure below. |project| defines a range of widgets, of which three are relevant for this example. The widget in orange contains a variety of methods of filter methods defined in :py:class:`tme.preprocessor.Preprocessor`. The blue widget in blue can be used to create a range of masks, perform axis-alignments and determine sensible default mask parameters. The green widget on the bottom right is used to export data to disk.

      .. figure:: ../_static/examples/napari_boot.png
         :width: 100 %
         :align: center

      Simply drag and drop your data in CCP4/MRC format into `napari` to import it. If your data is in a different format, you can convert it to CCP4/MRC from within Python using :py:meth:`tme.density.Density.from_file` and :py:meth:`tme.density.Density.to_file` like so:

      .. code-block:: python

         from tme import Density

         input_file = "/path/to/your/file"
         output_file = "output.mrc"

         Density.from_file(input_file).to_file(output_file)

      Make sure to adapt the paths ``input_file`` and ``output_file`` according to your specific use-case. Clicking the `Apply Filter` button will apply the filter to the target layer and create a new layer on the left hand side. The name of the new layer is indicative of the filter type used. The figure below shows the target used throughout this guide after application of a bandpass filter as outlined in the :ref:`Apply filters<preprocess-filtering>` section.

      .. figure:: ../_static/examples/napari_filter_widget.png
         :width: 100 %
         :align: center


      Note that the procedure is exactly the same for other targets. The figure below shows tomogram TS_037 (EMPIAR-10988, `click here <https://www.ebi.ac.uk/empiar/EMPIAR-10988/>`_) [3]_ before (left) and after application of a Gaussian filter (right).

      .. figure:: ../_static/examples/napari_filter_widget_tomogram.png
         :width: 100 %
         :align: center

      After selecting an adequate filter and parameter set the filtered data can be exported via the export button. Pressing the export button will open a separate window to determine the save location. If you applied a filter to the data, `napari` will also create a yaml file that contains the utilized parameters. Assuming your output is named `filtered.mrc`, the corresponding yaml file will be `filtered.yaml`.

      For the tomogram example above, the yaml file will contain the following:

      .. code-block:: yaml

         - gaussian_filter:
            sigma: 2.0

      The yaml file can be used in batch processing to apply the same filtering procedure to the new input `other_file.mrc` and write the output to `other_file_processed.mrc` for subsequent use.

      .. code-block:: bash

         preprocess.py \
            -i other_file.mrc \
            -y filtered.yml \
            -o other_file_processed.mrc


      The mask widget is operated anologously. In the following, we consider an averaged electron microscope map EMD-3228. We can increase the lower bound on the contrast in the `layer controls` widget and use the display mode toggle highlighted in blue, to obtain a more informative visualization of the map.

      .. figure:: ../_static/examples/napari_mask_widget_intro.png
         :width: 100 %
         :align: center

      The dropdown menu allows users to choose from a variety of masks with a given set of parameter. The `Adapy to layer` button will determine initial mask parameters based on the data of the currently selected layer. `Align to axis` will rotate the axis of largest variation in the selected layer onto the z-Axis, which can simplify mask creation for cases whose initial positioning is suboptimal. `Create mask` will create a specified mask for the currently selected layer.

      When clicking `Adapy to layer` followed by `Create mask`, we observe that the resulting sphere is excessively large. This is because EMD-3228 contains a fair amount of noisy unstructed density around the center structure. This can be assessed by reducing the lower bound on the contrast limit slider.

      .. figure:: ../_static/examples/napari_mask_widget_map.png
         :width: 100 %
         :align: center

      We can either adapt the mask manually, or make use of the Data Quantile feature. The Data Quantile allows us to only consider a given top percentage of the data for mask generation. In this case, 94.80 appeared to be a reasonable cutoff. Make sure to select the non-mask layer before clicking on `Adapy to layer`. The output is displayed below.

      .. figure:: ../_static/examples/napari_mask_widget_map2.png
         :width: 100 %
         :align: center

      Albeit more reasonable, the mask now cuts-off some parts of the map. We can mitigate this by manually fine-tuning the parameters until the mask at least encapsulates the most important features of the map.

      .. figure:: ../_static/examples/napari_mask_widget_map3.png
         :width: 100 %
         :align: center

      The generated masks can now be exported for subsequent use in template matching.

      Finally, we can also use the viewer to determine the wedge mask specification. For this, drag and drop a CCP4/MRC file for which you would expect a missing wedge into viewer. Select and apply the `Power Spectrum` filter to visualize the missing wedge. You might need to switch the image axis using the widgets on the bottom left. Once the wedge is visible, enable axis using View > Axes > Axes visible. Head over to the mask widget and select wedge. The opening axis corresponds to the axis that runs through the void defined by the wedge, the tilt axis is the axis the plane is tilted over. For the data in the example below, a tilt range of (40, 40), tilt axis 2 and opening axis 0 appears sufficient. ``match_template.py`` follows the same conventions, as you will see in subsequent tutorials.

      .. figure:: ../_static/examples/napari_mask_widget_wedge.png
         :width: 100 %
         :align: center

      .. note::

         Napari sometimes exits with a bus_error on MacOS systems. Usually its sufficient to reinstall napari and its dependencies, especially pyqt.


   .. tab-item:: API

      The :py:class:`tme.preprocessor.Preprocessor` class defines a range of filters that are outlined on the :doc:`../reference/preprocessor` page. :py:class:`tme.preprocessor.Preprocessor` operates on :obj:`numpy.ndarray` instances, such as the :py:attr:`tme.density.Density.data` attribute of :py:class:`tme.density.Density` instances. :py:meth:`tme.matching_utils.create_mask` on the other hand can be used to define a variety of masks for template matching.

      The following outlines the creation of an ellipsoidal mask and its subsequent smoothing. For further usage examples please have a look at the code attached to the figures in :ref:`preprocess-filtering` and :ref:`mask-design`.

      In the following we are going to use :py:meth:`tme.matching_utils.create_mask` to create an ellipsoid, in this case a sphere, with radius 15 as input. :py:meth:`tme.matching_utils.create_mask` returns a :obj:`numpy.ndarray` object, which we use in turn to create a :py:class:`tme.density.Density` instance. Note that by default, the instances attributes :py:attr:`tme.density.Density.sampling_rate` will be initialized to one unit per voxel, and the :py:attr:`tme.density.Density.origin` attribute to zero. We can write the created mask to disk using :py:meth:`tme.density.Density.to_file`.

      .. code-block:: python

         from tme import Density
         from tme.matching_utils import create_mask

         mask = create_mask(
            mask_type = "ellipse",
            center = (25, 25, 25),
            shape = (50, 50, 50),
            radius = (15, 15, 15)
         )
         Density(mask).to_file("example.mrc")

      In the following we will proceed with the ``example.mrc`` file we generated. However, you can adapt this to a file of your choice, as long as it can be read by :py:meth:`tme.density.Density.from_file`. We can apply a Gaussian filter with standard deviation ``sigma`` using :py:class:`tme.preprocessor.Preprocessor.gaussian_filter` and write the output to disk as ``example_gaussian_filter.mrc`` like so:

      .. code-block:: python

         from tme import Density, Preprocessor

         preprocessor = Preprocessor()
         dens = Density.from_file("example.mrc")

         out = dens.empty
         out.data = preprocessor.gaussian_filter(
            template = dens.data, sigma = 2
         )
         out.to_file("example_gaussian_filter.mrc")

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
         axs[0].set_title("Spherical Mask", color="#0a7d91")
         axs[1].set_title("Spherical Mask + Gaussian Filter", color="#0a7d91")

      :py:class:`tme.preprocessor.Preprocessor` supports a range of other operations such as the creation of wedge masks. The following outlines the creation of a continuous wedge mask assuming an infinite plane using :py:meth:`tme.preprocessor.Preprocessor.continuous_wedge_mask`.

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


Conclusion
----------

Following this guide, you have gained understand of data preprocessing within the context of template matching. Specifically:

- Learned the background and implications of filtering.

- Assessed how mask design shapes template matching scores.

- Utilitzed preprocessing techniques from within the `napari` GUI and |project|'s API.

In the subsequent tutorial, you will learn how to put these lessons to use when template matching on biological data.

.. _References:

References
----------

.. [1] Briechle, K.; Hanebeck, U. D. Template matching using fast normalized cross correlation. Optical Pattern Recognition XII. 2001; pp 95–102.
.. [2] Maurer, V. J.; Siggel, M.; Kosinski, J. The Shape of Things in Cryo-ET: Why Emojis Aren’t Just for Texts. bioRxiv 2023
.. [3] de Teresa-Trueba, I.; Goetz, S. K.; Mattausch, A.; Stojanovska, F.; Zimmerli, C. E.; Toro-Nahuelpan, M.; Cheng, D. W. C.; Tollervey, F.; Pape, C.; Beck, M.; Diz-Munoz, A.; Kreshuk, A.; Mahamid, J.; Zaugg, J. B. Convolutional networks for supervised mining of molecular patterns within cellular context. Nat. Methods 2023, 20, 284–294.
.. [4] Grant, T.; Rohou, A.; Grigorieff, N. cisTEM, user-friendly software for single-particle image processing. eLife 2018, 7, e35383.
.. [5] Padfield, D. Masked FFT registration. 2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition. 2010; pp 2918–2925
.. [6] Shujun Cai, C. Chen, Z. Y. Tan, Y. Huang, J. Shi, and L. Gan. Cryo-ET reveals the macromolecular reorganization of S. pombe mitotic chromosomes in vivo. Proc. Natl. Acad. Sci. 115, 10977 (2018).
.. [7] Stephan Nickell, O. Mihalache, F. Beck, R. Hegerl, A. Korinek, and W. Baumeister. Structural analysis of the 26S proteasome by cryoelectron tomography. Biochem. Biophys. Res. Commun. 353, 115 (2007).
.. [8] Misjael N. Lebbink, W. J. Geerts, T. P. van der Krift, M. Bouwhuis, L. O. Hertzberger, A. J. Verkleij, and A. J. Koster. Electron tomography & template matching of biological membranes. J. Struct. Biol. 158, 327 (2007).
