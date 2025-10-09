import scipy.io
from pathlib import Path
import torch

source_folder = Path('data') / 'behavioral_data_09052025' / 'mat'
destination_folder = Path('data') / 'behavioral_data_09052025' / 'pt'

source_data = source_folder.glob('**/*.mat')
for f_name in source_data:
    mat_data = torch.from_numpy(scipy.io.loadmat(f_name)['response_matrix'])
    str_fname = str(f_name).split('/')[-1].split('.')[0] + '.pt' # get raw file name and add new extension
    torch.save(mat_data, Path(destination_folder) / str_fname)

