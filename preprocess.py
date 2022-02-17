'''
PATH INITIALIZATION

PATH = Origninal Audio Files
SAVE_PATH = Transformed Audio Files
META_DATA_PATH = Metadata.json location
'''
import os

import librosa

import random
import numpy as np
import soundfile as sf
import multiprocessing as mp
import tqdm
import json

PATH = "/Users/valleotb/Desktop/Valleotb/sample"
SAVE_PATH = "/Users/valleotb/Desktop/Valleotb/sample_save"
META_DATA_PATH = '/Users/valleotb/Desktop/Valleotb/sample_metadata'
# FILE_COUNT = get_files_count(PATH)
FILE_LIST = os.listdir(PATH)
# for f in FILE_LIST:
#     if f.endswith('.flac'):
#         FILE_LIST.remove(f)

'''
NAMING SETTING
'''
j = 100000  # 원본 파일 네이밍
z = 200000  # 수정 후 파일 네이밍

'''
PARAMETER SETTING
'''
SAMPLE_TIME = 16000 * 3 # (16000 * second)
OUTPUT_DIMENSION = 10

'''
META DATA INITIALIZATION
'''
META_DATA = f"{META_DATA_PATH}/metadata.json"
DATA = {}

# DATA['speech_segments'] = []

# for i in range(FILE_COUNT - 1):
# for fname in FILE_LIST:
#
#     '''
#     AUDIO PREPROCESSING (CUT TO SAME TIMES)
#     '''
#     # WAVEFORM, SR = sf.read(f"{PATH}/{j}.flac")

def preprocess_single_data(fname):
    try:
        WAVEFORM, SR = sf.read(os.path.join(PATH, fname))
    except RuntimeError:
        return {fname: None}
    WAVEFORM = librosa.util.normalize(WAVEFORM)
    NON_SILENT_TIME = librosa.effects.split(WAVEFORM, top_db=20)

    WAVEFORM = WAVEFORM[NON_SILENT_TIME[0][0] : (NON_SILENT_TIME[0][0]+SAMPLE_TIME)]
    NON_SILENT_TIME_2 = librosa.effects.split(WAVEFORM, top_db=15)
    # print(NON_SILENT_TIME_2)

    if len(NON_SILENT_TIME_2) > 2 :
        random_num = random.randint(0, len(NON_SILENT_TIME_2)-1)
        k = NON_SILENT_TIME_2[random_num]
        if (k[1] - k[0]) > (SAMPLE_TIME / OUTPUT_DIMENSION) :

            '''
            DATA TRANSFORM PROCESS
            '''
            # Pitch Shift Process
            N_STEPS = random.randint(1,5)
            WAVEFORM_TRANSFORM = librosa.effects.pitch_shift(WAVEFORM[k[0]:k[1]], sr=SR, n_steps=N_STEPS)
            WAVEFORM_TRANSFORM_MINUS = librosa.effects.pitch_shift(WAVEFORM_TRANSFORM, sr=SR, n_steps=-N_STEPS)

            # Normalize
            WAVEFORM_NOT_TRANSFORM_PART = WAVEFORM[k[0]:k[1]]
            WAVEFORM_DOUBLED = WAVEFORM_NOT_TRANSFORM_PART ** 2
            WAVEFORM_TRANSFORM_DOUBLED = WAVEFORM_TRANSFORM_MINUS ** 2
            a = np.sqrt((WAVEFORM_DOUBLED.sum()) / (WAVEFORM_TRANSFORM_DOUBLED.sum()))

            WAVEFORM_TRANSFORM_FINAL = WAVEFORM_TRANSFORM_MINUS * a
            WAVEFORM[k[0]:k[1]] = WAVEFORM_TRANSFORM_FINAL

            '''
            DATA LABELING PROCESS

            LABELING => 구간의 50% 이상이 포함되었을때만 Labeling 진행한다.
            LABEL_DATA, POINTER -> Initializing
            '''
            LABEL_DATA = []
            POINTER = 0

            # sf.write(f'{SAVE_PATH}/{z}.wav', WAVEFORM, SR, subtype="PCM_24")
            save_fname = os.path.join(SAVE_PATH, fname.split('.')[0] + '.wav')
            sf.write(save_fname, WAVEFORM, SR, subtype="PCM_24")
            for e in range(OUTPUT_DIMENSION):
                if POINTER + (SAMPLE_TIME / OUTPUT_DIMENSION) <= k[0]:
                    # save label
                    LABEL_DATA.append(0)
                    POINTER += (SAMPLE_TIME / OUTPUT_DIMENSION)     # POINTER += 4800

                elif k[0] < POINTER + (SAMPLE_TIME / OUTPUT_DIMENSION) <= k[1]:
                    # save label
                    LABEL_DATA.append(1)
                    POINTER += (SAMPLE_TIME / OUTPUT_DIMENSION)     # POINTER += 4800

                elif POINTER + (SAMPLE_TIME / OUTPUT_DIMENSION) > k[1]:
                    # save label
                    LABEL_DATA.append(0)
                    POINTER += (SAMPLE_TIME / OUTPUT_DIMENSION)     # POINTER += 4800

            # # write metadata
            # DATA[f"{z}.wav"] = LABEL_DATA

            # process end
            # print(f'complete{z}')
            # z += 1
            return {os.path.basename(save_fname): LABEL_DATA}

if __name__ == "__main__":
    with mp.Pool(processes=8) as pool:
        metadata_list = list(
            tqdm.tqdm(
                pool.imap_unordered(
                    preprocess_single_data, FILE_LIST,
                ),
                total=len(FILE_LIST),
                desc=f'preprocessing in progress'
            )
        )
    print(len(metadata_list))
    for dic in metadata_list:
        if dic is not None:
            DATA.update(dic)
    with open(META_DATA, 'w') as outfile:
        json.dump(DATA, outfile)