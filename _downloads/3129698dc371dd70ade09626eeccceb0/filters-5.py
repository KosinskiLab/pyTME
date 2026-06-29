from tme import Density
from tme.filters import WedgeReconstructed

wedge = WedgeReconstructed(
    angles = [60,60],
    opening_axis = 0,
    tilt_axis = 1,
    create_continuous_wedge = True,
    weight_wedge = False,
    reconstruction_filter = "cosine"
)

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 5), constrained_layout=True)
for ax in axs.flat:
    ax.axis("off")

mask = wedge(shape = (100,100), return_real_fourier = False)["data"]
axs[0].imshow(np.fft.fftshift(mask), cmap = "gray")
axs[0].set_title('Continuous Wedge', color = '#24a9bb')

wedge.create_continuous_wedge = False
wedge.angles = np.linspace(-60, 60, 20)
mask = wedge(shape = (100,100), return_real_fourier = False)["data"]
axs[1].imshow(np.fft.fftshift(mask), cmap = "gray")
axs[1].set_title('Discrete Wedge', color = '#24a9bb')

wedge.weight_wedge = True
mask = wedge(shape = (100,100), return_real_fourier = False)["data"]
axs[2].imshow(np.fft.fftshift(mask), cmap = "gray")
axs[2].set_title('Weighted Discrete Wedge', color = '#24a9bb')

plt.show()