from pathlib import PurePath
from pydub import AudioSegment
import os
# This file is created for conversion for dataset Librispeech and vctk or any other audio format for non wav form

# revised based on the answer of https://stackoverflow.com/questions/58424108/with-python-how-can-i-convert-a-flac-stream-to-a-wav-stream
# This code is used on colab,  for the people want to implement on colab, still need the authencation  process

#Zhihong 
# This method is used for convert file from flac to wav








# alternate way flac_tmp_audio_data.export(file_path.with_suffix(".wav"))

def convert_audio(filename):
    file_path = PurePath("audio.flac")
    flac_tmp_audio_data = AudioSegment.from_file(file_path, file_path.suffix[1:])
    flac_tmp_audio_data.export(file_path.name.replace(file_path.suffix, "") + ".wav", format="wav")
# alternate way flac_tmp_audio_data.export(file_path.with_suffix(".wav"))

 

if __name__ == '__main__':
    foldername=r'F:/198/';  #folder name
    all_files = os.listdir(foldername) #get all filenames
    audio_files =  [ filename for filename in all_files if filename.endswith('.flac') ] #get .wav filenames

    for audio_file in audio_files: #loop over audio files
       a=convert_audio(filename)
  