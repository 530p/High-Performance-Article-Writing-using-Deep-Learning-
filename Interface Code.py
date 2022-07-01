import os
import warnings
import re
import numpy as np
import pandas as pd
import seaborn as sns
from keras_gpt_2 import load_trained_model_from_checkpoint, get_bpe_from_files, generate
import gpt_2_simple as gpt2
import streamlit as st
import warnings
import base64
import time

import streamlit.components as stc
timestr = time.strftime("%Y%m%d-%H%M%S")


class FileDownloader(object):

    def __init__(self, data, filename='myfile', file_ext='txt'):
        super(FileDownloader, self).__init__()
        self.data = data
        self.filename = filename
        self.file_ext = file_ext

    def download(self):
        b64 = base64.b64encode(self.data.encode()).decode()
        new_filename = "{}_{}_.{}".format(
            self.filename, timestr, self.file_ext)
        st.markdown("#### Download File ###")
        href = f'<a href="data:file/{self.file_ext};base64,{b64}" download="{new_filename}">Click Here!!</a>'
        st.markdown(href, unsafe_allow_html=True)


warnings.filterwarnings("ignore")
model_folder = 'G:/work gpt2/checkpoint/run1'
config_path = os.path.join(model_folder, 'hparams.json')
checkpoint_path = "G:/work gpt2/checkpoint/run1.ckpt"
encoder_path = os.path.join(model_folder, 'encoder.json')
vocab_path = os.path.join(model_folder, 'vocab.bpe')


st.header("High Performance Content Writer Using Deep Learning")
print('Load model from checkpoint...')
model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
print('Load BPE from files...')
bpe = get_bpe_from_files(encoder_path, vocab_path)
print('Generate text...')
text = st.text_input("Enter the Text")
length = st.number_input("Enter the Length", step=1)

if st.button('Click to run the content writer'):
    output = generate(
        model, bpe, [text], length, top_k=40, temperature=0.7)
    st.write(output[0])
    download = FileDownloader(output[0]).download()
