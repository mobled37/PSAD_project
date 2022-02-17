import os

TARGET_ROOT = '/Users/valleotb/Desktop/Valleotb/sample'
SRC_ROOT = "/Users/valleotb/Downloads/LibriSpeech4/train-clean-360"


def transfer_one_flac(directory):
    if os.path.isdir(directory):
        for d in os.listdir(directory):
            transfer_one_flac(os.path.join(directory, d))
    else:
        os.system(f'sudo cp {directory} {os.path.join(TARGET_ROOT)}')


def transfer_flacs():
    dir_list = os.listdir(SRC_ROOT)
    for directory in dir_list:
        transfer_one_flac(os.path.join(SRC_ROOT, directory))

if __name__ == "__main__":
    transfer_flacs()
