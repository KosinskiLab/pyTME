.. include:: ../../substitutions.rst

=======
Masking
=======

The template mask is pivotal for the normalization of the cross-correlation coefficient. The composition of an optimal mask is problem-specific, but we recommend the following when designing masks

1. The mask should focus on conserved key features.

2. Consider the context in which the mask will be applied. Particularly for *in situ* images, crowdedness can be problematic if the mask is excessively large.

3. Sharp transitions or edge effects should be avoided by smoothing or sufficiently sized boxes.

Next to template masks |project| also supports masks for the target, the integration of which depends on the used score. Typically however, target masks are used to remove regions rich in contaminants.


Mask Design
-----------

While filtering typically has greater impact on the performance than the choice of mask, chosing a suitable mask still helps to improve performance.

By default, |project| uses the entire area under the template as mask. However, this results in wide peaks, which is disadvantageous in corwded settings. When focusing on more defining features using an ellipsoidal mask we observe a sharpening of the peak and a decrease in background score. However, if we make the mask too narrow, as illustrated by the last example, the mask no longer provides a faithful approximation of the cross-correlation background and as thus the template matching scores become uninformative.

.. plot::
    :caption: Influence of mask design on template matching scores.

    import copy

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
        axs[1, 0].imshow(score / score.max())
        axs[0, 0].set_title("Default Mask", color="#24a9bb")
        axs[1, 0].set_title("Score", color="#24a9bb")

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
        axs[0, 1].imshow(masked_template, cmap=colormap)
        axs[1, 1].imshow(score / score.max())
        axs[0, 1].set_title("Ellipsoidal Mask", color="#24a9bb")
        axs[1, 1].set_title("Score", color="#24a9bb")

        template_mask = create_mask(
            mask_type="ellipse", radius=(5, 5), shape=template.shape, center=mask_center
        )
        score = compute_score(
            target=target,
            template=template,
            template_mask=template_mask,
            score=matching_score,
        )
        masked_template = template.copy()
        masked_template[template_mask == 0] = np.nan
        axs[0, 2].imshow(masked_template, cmap=colormap)
        axs[1, 2].imshow(score / score.max())
        axs[0, 2].set_title("Spherical Mask", color="#24a9bb")
        axs[1, 2].set_title("Score", color="#24a9bb")

        plt.show()

Mask Smoothing
--------------

The masks showed in the previous section contained sharp edges, which are troublesome to represent in Fourier space. In the following we look at the effect of smoothing the ellipsoidal mask using a Gaussian filter.

Albeit difficult to see in this representation, smoothing the mask with a sigma of two results in 5% higher peaks, compared to their individual backgrounds. Nevertheless, the benefit of smoothing masks is less obvious and has to be evaluated on a per-problem basis.

.. plot::
    :caption: Influence of mask smoothing on template matching scores.

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


.. note::

   The choice of mask also has an impact on the runtime performance of template matching. Rotation symmetric masks require less Fourier tarnsforms, thus reducing runtime by up to a factor of three.

   A more complete treatment of the mathematical implications of using different masks is provided in [1]_, together with `explanatory slides <https://pdfs.semanticscholar.org/17e5/419d9eb239b91b46fde52538e6c13b33909a.pdf>`_. Alternatively, `this OpenCV tutorial <https://docs.opencv.org/3.4/de/da9/tutorial_template_matching.html>`_ provides further examples on the effect of masking.



References
----------

.. [1] Padfield, D. Masked FFT registration. 2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition. 2010; pp 2918–2925