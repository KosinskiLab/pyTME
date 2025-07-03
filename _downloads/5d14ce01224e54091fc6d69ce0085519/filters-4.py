import copy
import numpy as np
import matplotlib.pyplot as plt

from tme import Density
from tme.filters import CTFReconstructed

target = Density.from_file("../../_static/examples/preprocessing_target.png")

ctf = CTFReconstructed(
    shape=target.shape,
    sampling_rate=target.sampling_rate[0],
    acceleration_voltage=200 * 1e3,
    defocus_x=[1000],
    spherical_aberration=2.7e7,
    amplitude_contrast=0.08,
    flip_phase=False,
    return_real_fourier=True,
)

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 5), constrained_layout=True)
for ax in axs.flat:
    ax.axis("off")

ctf_mask = ctf()["data"]
target_filtered = np.fft.irfftn(np.fft.rfftn(target.data) * ctf_mask)
axs[0].imshow(target_filtered, cmap="gray")
axs[0].set_title("Defocus 1000", color="#24a9bb")

ctf.defocus_x[0] = 2500
ctf_mask = ctf()["data"]
target_filtered = np.fft.irfftn(np.fft.rfftn(target.data) * ctf_mask)
axs[1].imshow(target_filtered, cmap="gray")
axs[1].set_title("Defocus 2500", color="#24a9bb")

ctf.defocus_x[0] = 5000
ctf_mask = ctf()["data"]
target_filtered = np.fft.irfftn(np.fft.rfftn(target.data) * ctf_mask)
axs[2].imshow(target_filtered, cmap="gray")
axs[2].set_title("Defocus 5000", color="#24a9bb")
plt.show()