import os
import os.path
import torch
import json
from torch.utils.data import Dataset
import torchaudio
from torch.utils.data.dataloader import DataLoader
import multiprocessing
import tqdm

class PSAD_Dataset(Dataset):

    def __init__(self,
                 audio_folder_dir,
                 metadata_dir,
                 device, load_first: bool = False):

        self.audio_folder_dir = audio_folder_dir
        metadata_path = metadata_dir
        self.meta_data_json = json.load(open(metadata_path))
        self.wav_path_list = list(self.meta_data_json.keys())

        self.device = device
        self.load_first = load_first
        if load_first:
            self.wav_list = {}
            with multiprocessing.Pool(processes=8) as pool:
                wav_dict_list = list(
                    tqdm.tqdm(
                        pool.imap_unordered(self.load, self.wav_path_list),
                        total=len(self.wav_path_list),
                        desc='pre-loading wav data'
                    )
                )
            for dic in wav_dict_list:
                self.wav_list.update(dic)

    @staticmethod
    def load(fpath):
        y, sr = torchaudio.load(fpath)
        return {id: y}

    def __len__(self):
        # return len(json.load(open(f'{self.metadata_dir}')))
        return len(self.wav_path_list)

    def __getitem__(self, index):
        file_name = self._get_file_name(index)
        audio_path = self._get_audio_file_path(file_name)
        if self.load_first:
            signal = self.wav_list[audio_path]
        else:
            signal, sr = torchaudio.load(audio_path)
        signal = signal.to(self.device)
        label = self._get_audio_label(index, self.wav_path_list)
        return signal, torch.Tensor([label])

    def _get_audio_file_path(self, file_name):
        # path = os.path.join(f'{self.audio_folder_dir}/{file_name}')
        path = os.path.join(f'{file_name}')
        return path

    def _get_file_name(self, index):
        # meta_dict = list(meta_data_json.keys())
        # return meta_dict[index]
        return self.wav_path_list[index]

    def _get_audio_label(self, index, wav_path_list):
        # meta_data_json = json.load(open(metadata_dir))
        # meta_dict = list(meta_data_json.keys())
        return self.meta_data_json[wav_path_list[index]]

    def get_files_count(self, folder_path):
        dirlisting = os.listdir(folder_path)
        return len(dirlisting)



if __name__ == "__main__":
    FILENAME_DIR = '/Users/valleotb/Desktop/Valleotb/sample_metadata/metadata.json'
    AUDIO_DIR = '/Users/valleotb/Desktop/Valleotb/sample_save'

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"using device: {device}")

    dataset = PSAD_Dataset(
        audio_folder_dir=AUDIO_DIR,
        metadata_dir=FILENAME_DIR,
        device=device
    )

    print(f"There are {len(dataset)} samples in the dataset.")
    print(dataset)

    temp_loader = DataLoader(dataset=dataset, batch_size=4, shuffle=False, drop_last=False)
    # print(PSAD_Dataset)
    print(next(iter(temp_loader)))
