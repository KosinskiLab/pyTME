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