import copy

import numpy as np
import matplotlib.pyplot as plt

from tme import Density

target = Density.from_file("../_static/examples/preprocessing_target.png").data
target = target.astype(np.float32)
template = Density.from_file("../_static/examples/preprocessing_template.png")
template.data = template.data.astype(np.float32)
template.pad(new_shape = target.shape, center = True, padding_value = np.nan)
template = template.data

fig, axs = plt.subplots(
   nrows = 1,
   ncols = 2,
   sharex=True,
   sharey=True,
   figsize = (12,5)
)

colormap = copy.copy(plt.cm.gray)
colormap.set_bad(color='white', alpha=0)

axs[1].imshow(template, cmap=colormap)
axs[1].set_title('Template', color = '#0a7d91')
axs[1].axis('off')

axs[0].imshow(target, cmap=colormap)
axs[0].set_title('Target', color='#0a7d91')
axs[0].axis('off')

plt.tight_layout()
plt.show()