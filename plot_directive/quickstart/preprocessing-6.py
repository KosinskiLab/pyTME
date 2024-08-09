import copy

import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import match_template

from tme import Density
from tme.preprocessing import LinearWhiteningFilter

target = Density.from_file("../_static/examples/preprocessing_target.png").data
template = Density.from_file(
   "../_static/examples/preprocessing_template.png"
).data

whitening_filter = LinearWhiteningFilter()
target_filter = whitening_filter(data = target)["data"]
template_filter = whitening_filter(data = template)["data"]

target_filtered = np.fft.irfftn(np.fft.rfftn(target) * target_filter)
template_filtered = np.fft.irfftn(np.fft.rfftn(template) * template_filter)

fig, axs = plt.subplots(nrows = 1, ncols = 1)

axs.imshow(match_template(
   target_filtered, template_filtered, pad_input = True)
)
axs.set_title('Template Matching Score (Spectral Whitening)', color = '#0a7d91')

plt.tight_layout()
plt.show()