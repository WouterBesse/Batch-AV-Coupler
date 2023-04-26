import librosa
import os
import numpy as np
from scipy import signal

AUDIO_EXTENSIONS = [
    '.wav', '.mp3', '.flac', '.sph', '.ogg', '.opus',
    '.WAV', '.MP3', '.FLAC', '.SPH', '.OGG', '.OPUS',
]


def is_audio_file(
        filename):  # is_audio_file and load_wav from https://github.com/swasun/VQ-VAE-Speech/blob/master/src/dataset/vctk.py
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)

def couple_av(v_path, audio_path, audio_paths, samplerate, window):
    video_audio, _ = librosa.load(v_path, sr=samplerate)
    within_video = True
    similarities = []
    similarities_t = []
    for a_path in audio_paths:
        a_path = os.path.join(audio_path, a_path)

        if not is_audio_file(a_path):
            continue

        recording, _ = librosa.load(a_path, sr=samplerate)

        if np.shape(recording)[-1] >= np.shape(video_audio)[-1]:
            y_within = recording
            y_find = video_audio
            within_video = False
        else:
            y_within = video_audio
            y_find = recording
            within_video = True

        c = signal.correlate(y_within, y_find[:samplerate * window], mode='full', method='fft')
        peak_index = np.argmax(c)
        peak_value = np.amax(c)
        similarities.append(peak_value)
        similarities_t.append(peak_index)

    # similarities = np.array(similarities)
    # similarity_index = np.argmax(similarities)
    # similar_recording = librosa.load(audio_paths[similarity_index], sr=samplerate)
    # overlap_index = similarities_t[similarity_index]
    return similarities, similarities_t

def batch_couple(video_path, audio_path, samplerate, window):
    video_paths = os.listdir(video_path)
    audio_paths = os.listdir(audio_path)

    audio_simvals = []
    audio_simidx = []

    for v_path in video_paths:
        similarities, similarities_t = couple_av(os.path.join(video_path, v_path),
                                      audio_path,
                                      audio_paths,
                                      samplerate,
                                      window)
        audio_simvals.append(similarities)
        audio_simidx.append(similarities_t)

    if within_video:
        similar_recording = np.pad(similar_recording,
                                   (overlap_index, 0),
                                   constant_values=0)  # Pad recording with 0's on the left to sync
        similar_recording = similar_recording[:, :np.shape(video_audio)[-1]]  # Crop audio to length of video audio
    else:
        similar_recording = np.pad(similar_recording,
                                   (overlap_index, 0),
                                   constant_values=0)  # Pad recording with 0's on the left to sync
        similar_recording = similar_recording[:, :np.shape(video_audio)[-1]]  # Crop audio to length of video audio





