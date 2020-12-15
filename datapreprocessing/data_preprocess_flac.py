#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Modified from https://github.com/JanhHyun/Speaker_Verification
import glob
import os
import numpy as np
import librosa
import soundfile as sf

#tried to use for Librispeech file

train_path = './train_tisv'
test_path  = './test_tisv'

# for convenience ,define parameter for hp term
tisv_frame= 180
hop = 0.01
window =0.025 #s
#train_path_unprocessed= './TIMIT/TRAIN/*/*/*.wav'
train_path_unprocessed= 'C:/Users/zz/Downloads/LibriSpeech/train-clean-100/*/*/*.flac'
#test_path_unprocessed: './TIMIT/TEST/*/*/*.wav'
test_path_unprocessed: 'C:/Users/zz/Downloads/LibriSpeech/test-clean/*/*/*.flac'
nfft =512
nmels =40


# downloaded dataset path
audio_path = glob.glob(os.path.dirname("C:/Users/zz/Downloads/LibriSpeech/*/*/*/*.flac"))                                        

def save_spectrogram_tisv():
    """ Full preprocess of text independent utterance. The log-mel-spectrogram is saved as numpy file.
        Each partial utterance is splitted by voice detection using DB
        and the first and the last 180 frames from each partial utterance are saved. 
        Need : utterance data set (VTCK)
    """
    sr=16000
    print("start text independent utterance feature extraction")
    os.makedirs(train_path, exist_ok=True)   # make folder to save train file
    os.makedirs(test_path, exist_ok=True)    # make folder to save test file

    utter_min_len = (tisv_frame * hop + window) * sr    # lower bound of utterance length
    total_speaker_num = len(audio_path)
    train_speaker_num= (total_speaker_num//10)*9            # split total data 90% train and 10% test
    print("total speaker number : %d"%total_speaker_num)
    print("train : %d, test : %d"%(train_speaker_num, total_speaker_num-train_speaker_num))
    for i, folder in enumerate(audio_path):
        print("%dth speaker processing..."%i)
        utterances_spec = []
        for utter_name in os.listdir(folder):
            if utter_name[-4:] == '.flac':
                utter_path = os.path.join(folder, utter_name)         # path of each utterance
               # utter, sr = soundfile.load(utter_path, sr)        # load utterance audio
                utter, sr = sf.read(utter_path, samplerate)        # load utterance audio
                intervals = librosa.effects.split(utter, top_db=30)         # voice activity detection 
                # this works fine for timit but if you get array of shape 0 for any other audio change value of top_db
                # for vctk dataset use top_db=100
                for interval in intervals:
                    if (interval[1]-interval[0]) > utter_min_len:           # If partial utterance is sufficient long,
                        utter_part = utter[interval[0]:interval[1]]         # save first and last 180 frames of spectrogram.
                        S = librosa.core.stft(y=utter_part, n_fft=nfft,
                                              win_length=int(window * sr), hop_length=int(hop * sr))
                        S = np.abs(S) ** 2
                        mel_basis = librosa.filters.mel(sr=sr, n_fft=nfft, n_mels=nmels)
                        S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances
                        utterances_spec.append(S[:, :tisv_frame])    # first 180 frames of partial utterance
                        utterances_spec.append(S[:, -tisv_frame:])   # last 180 frames of partial utterance

        utterances_spec = np.array(utterances_spec)
        print(utterances_spec.shape)
        if i<train_speaker_num:      # save spectrogram as numpy file
            np.save(os.path.join(train_path, "speaker%d.npy"%i), utterances_spec)
        else:
            np.save(os.path.join(test_path, "speaker%d.npy"%(i-train_speaker_num)), utterances_spec)


if __name__ == "__main__":
    save_spectrogram_tisv()
