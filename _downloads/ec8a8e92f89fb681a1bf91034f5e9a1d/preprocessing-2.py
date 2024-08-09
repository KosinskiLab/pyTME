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