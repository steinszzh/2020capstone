#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 20:44:10 2020

@author: zhihongz
"""



import os 
from pydub import AudioSegment


def convert_to_wav(dir_path):
    for file_path in os.listdir(dir_path):
        if file_path.split('.')[-1] != "wav":
            read_file = AudioSegment.from_file(os.path.join(dir_path,file_path), file_path.split('.')[-1])
            os.remove(os.path.join(dir_path,file_path))
            base_name = file_path.split('.')[:-1]
            # read_file = read_file.set_channels(8)
            # base_name = ".".join(base_name)
            read_file.export(os.path.join(dir_path,f"{base_name[0]}.wav"), format="wav")
            
            
if __name__ == '__main__':
    dir_path= './dev-clean/2078/142845'  # folder name
    all_files = os.listdir(dir_path)  # get all filenames # get .wav filenames
    conv= convert_to_wav(dir_path)
