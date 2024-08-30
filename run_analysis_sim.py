from utils import load_sim_data

def run_analysis(f_name):
    """
    Carry out analysis on Jiaming's simulated data.

    Inputs:
    -------
    f_name: String
        Full file name to the simulated data. 

    """
    
    # load simulated data
    data = load_sim_data(f_name)