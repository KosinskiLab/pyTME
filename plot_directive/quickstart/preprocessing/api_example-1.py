import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from tme import Density, Preprocessor
from tme.matching_utils import create_mask

preprocessor = Preprocessor()
mask = create_mask(
        mask_type = "ellipse",
        center = (25, 25, 25),
        shape = (50, 50, 50),
        radius = (15, 15, 15)
).astype(np.float32)

filtered_mask = preprocessor.gaussian_filter(
        template = mask, sigma = 2
)

mask = mask.max(axis = 0)
filtered_mask = filtered_mask.max(axis = 0)

fig, axs = plt.subplots(
        nrows=1, ncols=2, sharex=False, sharey=False, figsize=(12, 5)
)
colormap = copy.copy(plt.cm.viridis)
colormap.set_bad(color="white", alpha=0)
norm = colors.Normalize(vmin=0, vmax=1)

mask[mask < np.exp(-2)] = np.nan
filtered_mask[filtered_mask < np.exp(-2)] = np.nan

axs[0].imshow(mask, cmap=colormap, norm = norm)
axs[1].imshow(filtered_mask, cmap=colormap, norm = norm)
axs[0].set_title("Spherical Mask", color="#24a9bb")
axs[1].set_title("Spherical Mask + Gaussian Filter", color="#24a9bb")