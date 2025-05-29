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