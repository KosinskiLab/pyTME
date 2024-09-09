from skimage.feature import match_template

from tme import Density, Preprocessor
from tme.preprocessing import BandPassFilter

target = Density.from_file("../../_static/examples/preprocessing_target.png").data
target_ft = np.fft.fftshift(np.fft.fft2(target))
template = Density.from_file("../../_static/examples/preprocessing_template.png").data

preprocessor = Preprocessor()
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 5), constrained_layout=True)
for ax in axs.flat:
   ax.axis("off")
lowpass = BandPassFilter(
   lowpass = 5,
   highpass = None,
   sampling_rate = 1,
)(shape = target.shape, shape_is_real_fourier = False, return_real_fourier=False)["data"]
lowpass = np.fft.ifftshift(lowpass)
target_lowpass = np.fft.ifft2(np.fft.ifftshift(lowpass * target_ft)).real

highpass = BandPassFilter(
   lowpass = None,
   highpass = 4,
   sampling_rate = 1,
)(shape = target.shape, shape_is_real_fourier = False, return_real_fourier=False)["data"]
highpass = np.fft.fftshift(highpass)
target_highpass = np.fft.ifft2(np.fft.ifftshift(highpass * target_ft)).real

bandpass = BandPassFilter(
   lowpass = 5,
   highpass = 10,
   sampling_rate = 1
)(shape = target.shape, shape_is_real_fourier = False, return_real_fourier=False)["data"]
bandpass = np.fft.fftshift(bandpass)
target_bandpass = np.fft.ifft2(np.fft.ifftshift(bandpass * target_ft)).real

score = match_template(target_lowpass, template)
axs[0, 0].imshow(target_lowpass, cmap='gray')
axs[0, 0].set_title('Low-pass Filtered', color = '#24a9bb')
axs[1, 0].imshow(score / score.max())
axs[1, 0].set_title('Score (Low-pass)', color = '#24a9bb')

score = match_template(target_highpass, template)
axs[0, 1].imshow(target_highpass, cmap='gray')
axs[0, 1].set_title('High-pass Filtered', color = '#24a9bb')
axs[1, 1].imshow(score / score.max())
axs[1, 1].set_title('Score (High-pass)', color = '#24a9bb')

score = match_template(target_bandpass, template)
axs[0, 2].imshow(target_bandpass, cmap='gray')
axs[0, 2].set_title('Band-pass Filtered', color = '#24a9bb')
axs[1, 2].imshow(score / score.max())
axs[1, 2].set_title('Score (Band-pass)', color = '#24a9bb')

plt.show()