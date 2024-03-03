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
      pad_target_edges=True,
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