import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import match_template

from tme import Density, Preprocessor

target = Density.from_file("../_static/examples/preprocessing_target.png").data
template = Density.from_file("../_static/examples/preprocessing_template.png").data

target_ft = np.fft.fftshift(np.fft.fft2(target))

preprocessor = Preprocessor()
fig, axs = plt.subplots(3, 1, figsize=(12, 15), constrained_layout=True)

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

axs[0].imshow(match_template(target_lowpass, template))
axs[0].set_title('Template Matching Score (Low-pass)', color = '#0a7d91')

axs[1].imshow(match_template(target_highpass, template))
axs[1].set_title('Template Matching Score (High-pass)', color = '#0a7d91')

axs[2].imshow(match_template(target_bandpass, template))
axs[2].set_title('Template Matching Score (Band-pass)', color = '#0a7d91')

plt.show()