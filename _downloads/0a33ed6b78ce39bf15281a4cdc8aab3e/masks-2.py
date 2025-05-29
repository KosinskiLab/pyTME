import copy
import matplotlib.colors as colors

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
        pad_target_edges=True,
        job_schedule=(1,1),
    )
    return candidates[0]

if __name__ == "__main__":

    matching_score = "FLC"

    target = Density.from_file("../../_static/examples/preprocessing_target.png").data
    template = Density.from_file("../../_static/examples/preprocessing_template.png").data
    target = target.astype(np.float32)
    template = template.astype(np.float32)

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 5), constrained_layout=True)
    for ax in axs.flat:
        ax.axis("off")

    colormap = copy.copy(plt.cm.viridis)
    colormap.set_bad(color="white", alpha=0)
    norm = colors.Normalize(vmin=0, vmax=1)

    mask_center = np.add(
        np.divide(template.shape, 2).astype(int), np.mod(template.shape, 2)
    )
    template_mask = create_mask(
        mask_type="ellipse", radius=(20, 10), shape=template.shape, center=mask_center
    ) * 1.0
    score = compute_score(
        target=target,
        template=template,
        template_mask=template_mask,
        score=matching_score,
    )
    template_mask[template_mask < 1] = np.nan
    axs[0, 0].imshow(template_mask, cmap=colormap, norm = norm)
    axs[1, 0].imshow(score)
    axs[0, 0].set_title("Sigma 0", color="#24a9bb")
    axs[1, 0].set_title("Score", color="#24a9bb")

    template_mask = create_mask(
        mask_type="ellipse", radius=(20, 10), shape=template.shape, center=mask_center,sigma_decay=2
    )
    score = compute_score(
        target=target,
        template=template,
        template_mask=template_mask,
        score=matching_score,
    )
    template_mask[template_mask == 0] = np.nan
    axs[0, 1].imshow(template_mask, cmap=colormap, norm = norm)
    axs[1, 1].imshow(score)
    axs[0, 1].set_title("Sigma 2", color="#24a9bb")
    axs[1, 1].set_title("Score", color="#24a9bb")

    template_mask = create_mask(
        mask_type="ellipse", radius=(20, 10), shape=template.shape, center=mask_center,sigma_decay=5
    )
    score = compute_score(
        target=target,
        template=template,
        template_mask=template_mask,
        score=matching_score,
    )
    template_mask[template_mask == 0] = np.nan
    axs[0, 2].imshow(template_mask, cmap=colormap, norm = norm)
    axs[1, 2].imshow(score)
    axs[0, 2].set_title("Sigma 5", color="#24a9bb")
    axs[1, 2].set_title("Score", color="#24a9bb")

    plt.show()