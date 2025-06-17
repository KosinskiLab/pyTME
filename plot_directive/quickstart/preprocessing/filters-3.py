import copy
from tme import Density
from tme.cli import match_template
from tme.filters import LinearWhiteningFilter

target = Density.from_file("../../_static/examples/preprocessing_target.png").data
template = Density.from_file(
   "../../_static/examples/preprocessing_template.png"
).data

whitening_filter = LinearWhiteningFilter()
target_filter = whitening_filter(
   data=target,
   shape=target.shape,
   return_real_fourier=True
)["data"]
template_filter = whitening_filter(
   data=template,
   shape=template.shape,
   return_real_fourier=True,
)["data"]

target_filtered = np.fft.irfftn(np.fft.rfftn(target) * target_filter)
template_filtered = np.fft.irfftn(np.fft.rfftn(template) * template_filter)

fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(10, 5), constrained_layout=True)
for ax in axs.flat:
   ax.axis("off")

score = match_template(target_filtered, template_filtered)[0]
score = score - score.mean()
score[score < 0] = 0
score = score / score.max()

axs[0].imshow(target_filtered, cmap = "gray")
axs[0].set_title('Whitened', color = '#24a9bb')
axs[1].imshow(score)
axs[1].set_title('Score (Spectral Whitening)', color = '#24a9bb')

plt.show()