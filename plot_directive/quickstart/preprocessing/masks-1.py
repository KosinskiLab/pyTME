import copy

from tme import Density
from tme.cli import match_template
from tme.matching_utils import create_mask

if __name__ == "__main__":

    matching_score = "FLC"

    target = Density.from_file("../../_static/examples/preprocessing_target.png").data
    template = Density.from_file("../../_static/examples/preprocessing_template.png").data
    target = target.astype(np.float32)
    template = template.astype(np.float32)

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 5), constrained_layout=True)
    for ax in axs.flat:
        ax.axis("off")
    colormap = copy.copy(plt.cm.gray)
    colormap.set_bad(color='white', alpha=0)

    template_mask = np.ones_like(template)
    score = match_template(
        target=target,
        template=template,
        template_mask=template_mask,
        score=matching_score,
    )[0]
    masked_template  = template.copy()
    masked_template[template_mask == 0] = np.nan

    axs[0, 0].imshow(masked_template, cmap=colormap)
    axs[1, 0].imshow(score / score.max())
    axs[0, 0].set_title("Default Mask", color="#24a9bb")
    axs[1, 0].set_title("Score", color="#24a9bb")

    mask_center = np.add(
        np.divide(template.shape, 2).astype(int), np.mod(template.shape, 2)
    )
    template_mask = create_mask(
        mask_type="ellipse", radius=(20, 10), shape=template.shape, center=mask_center
    )

    score = match_template(
        target=target,
        template=template,
        template_mask=template_mask,
        score=matching_score,
    )[0]
    masked_template  = template.copy()
    masked_template[template_mask == 0] = np.nan
    axs[0, 1].imshow(masked_template, cmap=colormap)
    axs[1, 1].imshow(score / score.max())
    axs[0, 1].set_title("Ellipsoidal Mask", color="#24a9bb")
    axs[1, 1].set_title("Score", color="#24a9bb")

    template_mask = create_mask(
        mask_type="ellipse", radius=(5, 5), shape=template.shape, center=mask_center
    )
    score = match_template(
        target=target,
        template=template,
        template_mask=template_mask,
        score=matching_score,
    )[0]
    masked_template = template.copy()
    masked_template[template_mask == 0] = np.nan
    axs[0, 2].imshow(masked_template, cmap=colormap)
    axs[1, 2].imshow(score / score.max())
    axs[0, 2].set_title("Spherical Mask", color="#24a9bb")
    axs[1, 2].set_title("Score", color="#24a9bb")

    plt.show()