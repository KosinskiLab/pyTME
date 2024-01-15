import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from tme import Density

target = Density.from_file("../_static/examples/preprocessing_target.png").data
target_fft = np.fft.fftshift(np.fft.fft2(target))

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].imshow(target, cmap='gray')
axs[0].set_title('Target', color = '#0a7d91')

axs[1].imshow(np.log(np.abs(target_fft)), cmap='gray')
axs[1].set_title('Fourier Transform Magnitude', color = '#0a7d91')

radius = 50
center = np.divide(target_fft.shape, 2).astype(int)[::-1]

aspect_ratio = target_fft.shape[1] / target_fft.shape[0]

for i in range(1,4):
   radius_y = i * 50  # Radius in the x direction
   radius_x = radius_y * aspect_ratio  # Adjusted radius in the y direction
   ellipse = mpatches.Ellipse(center, 2*radius_x, 2*radius_y, color='red', fill=False)
   axs[1].add_patch(ellipse)


arrow_stop = center.copy()
arrow_stop[0] *= 1.75

arr = mpatches.FancyArrowPatch(
   center, arrow_stop,
   arrowstyle='->,head_width=.15',
   mutation_scale=20
)
axs[1].add_patch(arr)
axs[1].annotate(
   "Frequency ↑", (.5, .5), xycoords=arr, ha='center', va='bottom'
)

plt.tight_layout()
plt.show()