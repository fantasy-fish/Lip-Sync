'''
Foreground voice extraction for audio files
python3 audio_segmentation.py
'''
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os

def process(filename, new_filename):
	y, sr = librosa.load(filename)


	# And compute the spectrogram magnitude and phase
	S_full, phase = librosa.magphase(librosa.stft(y))

	# We'll compare frames using cosine similarity, and aggregate similar frames
	# by taking their (per-frequency) median value.
	#
	# To avoid being biased by local continuity, we constrain similar frames to be
	# separated by at least 2 seconds.
	#
	# This suppresses sparse/non-repetetitive deviations from the average spectrum,
	# and works well to discard vocal elements.

	S_filter = librosa.decompose.nn_filter(S_full,
	                                       aggregate=np.median,
	                                       metric='cosine',
	                                       width=int(librosa.time_to_frames(2, sr=sr)))

	# The output of the filter shouldn't be greater than the input
	# if we assume signals are additive.  Taking the pointwise minimium
	# with the input spectrum forces this.
	S_filter = np.minimum(S_full, S_filter)


	# We can also use a margin to reduce bleed between the vocals and instrumentation masks.
	# Note: the margins need not be equal for foreground and background separation
	margin_i, margin_v = 2, 10
	power = 2

	mask_i = librosa.util.softmask(S_filter,
	                               margin_i * (S_full - S_filter),
	                               power=power)

	mask_v = librosa.util.softmask(S_full - S_filter,
	                               margin_v * S_filter,
	                               power=power)

	# Once we have the masks, simply multiply them with the input spectrum
	# to separate the components

	S_foreground = mask_v * S_full
	S_background = mask_i * S_full


	import soundfile as sf
	new_y = librosa.istft(S_foreground*phase)
	sf.write(new_filename, new_y, samplerate=sr, subtype='PCM_24')

def get_image_list(data_root, split):
	sub_data_root = os.path.join(data_root, split)
	folder_list = os.listdir(sub_data_root)
	folder_list = [os.path.join(sub_data_root, folder) for folder in folder_list]
	return folder_list


if __name__ == '__main__':
	for split in ['train', 'test']:
		folder_list = get_image_list('av-toy-preprocessed', 'test')
		for folder in folder_list:
			print(folder)
			filename = os.path.join(folder, 'audio.wav')
			new_filename = os.path.join(folder, 'audio1.wav')
			process(filename, new_filename)