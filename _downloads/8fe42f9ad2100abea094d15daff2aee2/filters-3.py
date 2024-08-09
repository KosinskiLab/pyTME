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