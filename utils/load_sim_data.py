import scipy
import torch

def load_sim_data(f_name):
    """
    This function loads the simulated pairwise discriminality matrix (.mat file) and converts it into a torch tensor.  

    Input:
    -------
    f_name: String
        Full file name to the simulated data. 
        
    Output:
    --------
    data: (n_frames x n_frames) Torch array
        Matrix containing the pairwise discriminality matrix.

    """

    data_matlab = scipy.io.loadmat(f_name)
    data = torch.from_numpy(data_matlab['Pc_reshaped'])

    return data


