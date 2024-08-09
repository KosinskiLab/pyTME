import numpy as np
import matplotlib.pyplot as plt
from tme import Density, Preprocessor

target = Density.from_file("../_static/examples/preprocessing_target.png").data
target_ft = np.fft.fftshift(np.fft.fft2(target))

preprocessor = Preprocessor()
fig, axs = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)

lowpass = np.fft.fftshift(preprocessor.bandpass_mask(
   shape = target_ft.shape,
   minimum_frequency = 0,
   maximum_frequency = .1,
   omit_negative_frequencies = False,
   gaussian_sigma = 2
))
target_lowpass = np.fft.ifft2(np.fft.ifftshift(lowpass * target_ft)).real

highpass = np.fft.fftshift(preprocessor.bandpass_mask(
   shape = target_ft.shape,
   minimum_frequency = .2,
   maximum_frequency = .75,
   omit_negative_frequencies = False,
   gaussian_sigma = 2
))
target_highpass = np.fft.ifft2(np.fft.ifftshift(highpass * target_ft)).real

bandpass = np.fft.fftshift(preprocessor.bandpass_mask(
   shape = target_ft.shape,
   minimum_frequency = .1,
   maximum_frequency = .4,
   omit_negative_frequencies = False,
   gaussian_sigma = 2
))
target_bandpass = np.fft.ifft2(np.fft.ifftshift(bandpass * target_ft)).real

axs[0, 0].imshow(lowpass)
axs[0, 0].set_title('Low-pass Filter', color = '#0a7d91')
axs[0, 1].imshow(target_lowpass, cmap='gray')
axs[0, 1].set_title('Low-pass Filtered', color = '#0a7d91')

axs[1, 0].imshow(highpass)
axs[1, 0].set_title('High-pass Filter', color = '#0a7d91')
axs[1, 1].imshow(target_highpass, cmap='gray')
axs[1, 1].set_title('High-pass Filtered', color = '#0a7d91')

axs[2, 0].imshow(bandpass)
axs[2, 0].set_title('Band-pass Filter', color = '#0a7d91')
axs[2, 1].imshow(target_bandpass, cmap='gray')
axs[2, 1].set_title('Band-pass Filtered', color = '#0a7d91')

plt.show()