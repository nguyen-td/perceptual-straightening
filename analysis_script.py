import argparse
from datetime import date
import subprocess
import torch

from utils import ParamsDict
from modules import Params, PerceptualData, PerceptualDataNull
import pickle
import modules

def load_params(param_file):
    """ 
    Load parameters.     TODO: Test if this works.

    Inputs:
        param_file: Pickle file containing parameters.
    """

    with open(load_params, 'rb') as file: params = pickle.load(file)
    print(
        f'Using params from {load_params}. Any additional arguments were not parsed and will be ignored.'
    )
    
def set_params():
    """
    Set parameters.

    Outputs:
        params: Dictionary containing parameters for the analysis.
        metadata: Dictionary containing hardcoded metadata for the analysis, only used if seed > 0. If seed - 0, metadata = None.
        sequences: Set containing sequences, e.g., {'groundtruth'}
        subjects: Set containing the names of the subjects.
        condictions: Set containing the names of the conditions.
    """

    params = dict()

    # Set any general parameters. E.g., run ID, rng seed.
    params['general'] = {
        'run_id':   'A',
        'rng_seed': 1
    }

    # Set preprocessing parameters for perceptual data.
    params['preprocess'] = {
        'experimenter': 'yoon', # This originally was `model = 'data_yoon'`
        'max_deviation': 1,
        'min_deviation': 1,
        'load_dir':     'data/pt'
    }

    # Set parameters for curvature computation. Mainly path to stimuli directory.
    params['curvature'] = {
        'load_dir': 'stimuli/stimuli'
    }

    # Set analysis parameters. 
    params['inference'] = {
        'seed':                 1,
        'dim':                  10,
        'mb':                   6, 
        'lr':                   0.01,
        'exp_averaging':        0.999,
        'max_iter':             8e4,
        'collect_losses_from':  5e4,
        'theta_transfer':       'sigmoid_identity',
        'dist_transfer':        'sigmoid10',
        'dist_prior':           'scalar', 
        'theta_prior':          'scalar',
        'acc_prior':            'zero_mean', 
        'z_post':               'full',
        'dist_init':            5,
        'repeat_number':        1, # previously: args.domain
        'nat_only':             True,
        'max_deivation':        2,
        'data_lapse':           0.12,
        'data_acc':             'orig_really',
        'min_lapse':            0,
        'max_lapse':            0.12,
        'lapse_dim':            2,
        'lapse_transfer':       'cdf_normal',
        'tags':                 ['big'] # XXX: Original Lua code had {'big'}, which is a table 
    }

    # Set parameters for bootstrapping.
    params['boot'] = {
        # ------------------------------------
        'boot_method':  'nonparam',
        'n_bootstraps': 100,
        'parallel':     True
    }

    # Set parameters for synthesis of null data.
    params['null'] = {}


    # What is this section doing? Fill in comment.
    if params['preprocess']['experimenter'] == 'yoon': 
        sequences  = {'groundtruth'}
        # subjects   = {'qj', 'yb'} 
        subjects   = {'alexandra', 'carlos', 'maddy', 'ryan'}
        conditions = {
            'natural_parafovea_04_EGOMOTION', 
            'natural_periphery_06_DAM', 
            'synthetic_fovea_06_DAM', 
            'synthetic_parafovea_04_EGOMOTION', 
            'natural_parafovea_05_PRAIRIE', 
            'synthetic_fovea_04_EGOMOTION', 
            'natural_fovea_05_PRAIRIE', 
            'natural_parafovea_04_EGOMOTION', 
            'synthetic_fovea_06_DAM', 
            'synthetic_parafovea_05_PRAIRIE', 
            'natural_fovea_05_PRAIRIE', 
            'natural_parafovea_04_EGOMOTION', 
            'natural_periphery_05_PRAIRIE', 
            'natural_periphery_06_DAM', 
            'synthetic_fovea_06_DAM', 
            'synthetic_parafovea_04_EGOMOTION', 
            'synthetic_periphery_05_PRAIRIE'
        }
    elif params['preprocess']['experimenter'] == 'corey':
        sequences  = {'groundtruth'}
        subjects   = {'cmz', 'CMZ'}
        conditions = {
            'pilot_movie1', 
            'pilot_movie2', 
            'PredControl_carnegie-dam', 
            'PredControl_leaves-wind', 
            'PredControl_ice3Mod', 
            'PredControl_beesMod', 
            'PredControl_water', 
            'PredControl_butterflies0', 
            'PredControl_prairie1Con', 
            'PredControl_chironomusMod'
        }
    else:
        raise NameError("params['preprocess']['experimenter'] must be 'yoon' or 'corey'")

    if params['seed'] > 0:
        # Values for Yoon were hard coded in Olivier's original implementation.
        if params['preprocess']['experimenter'] == 'yoon':
            metadata = {
            {'sequence' : 'groundtruth', 'subject' : 'alexandra', 'condition' : 'natural_parafovea_04_EGOMOTION'}, 
            {'sequence' : 'groundtruth', 'subject' : 'alexandra', 'condition' : 'natural_periphery_06_DAM'}, 
            {'sequence' : 'groundtruth', 'subject' : 'alexandra', 'condition' : 'synthetic_fovea_06_DAM'}, 
            {'sequence' : 'groundtruth', 'subject' : 'alexandra', 'condition' : 'synthetic_parafovea_04_EGOMOTION'}, 
            {'sequence' : 'groundtruth', 'subject' : 'carlos', 'condition' : 'natural_parafovea_05_PRAIRIE'}, 
            {'sequence' : 'groundtruth', 'subject' : 'carlos', 'condition' : 'synthetic_fovea_04_EGOMOTION'}, 
            {'sequence' : 'groundtruth', 'subject' : 'maddy', 'condition' : 'natural_fovea_05_PRAIRIE'}, 
            {'sequence' : 'groundtruth', 'subject' : 'maddy', 'condition' : 'natural_parafovea_04_EGOMOTION'}, 
            {'sequence' : 'groundtruth', 'subject' : 'maddy', 'condition' : 'synthetic_fovea_06_DAM'}, 
            {'sequence' : 'groundtruth', 'subject' : 'maddy', 'condition' : 'synthetic_parafovea_05_PRAIRIE'}, 
            {'sequence' : 'groundtruth', 'subject' : 'ryan', 'condition' : 'natural_fovea_05_PRAIRIE'}, 
            {'sequence' : 'groundtruth', 'subject' : 'ryan', 'condition' : 'natural_parafovea_04_EGOMOTION'}, 
            {'sequence' : 'groundtruth', 'subject' : 'ryan', 'condition' : 'natural_periphery_05_PRAIRIE'}, 
            {'sequence' : 'groundtruth', 'subject' : 'ryan', 'condition' : 'natural_periphery_06_DAM'}, 
            {'sequence' : 'groundtruth', 'subject' : 'ryan', 'condition' : 'synthetic_fovea_06_DAM'}, 
            {'sequence' : 'groundtruth', 'subject' : 'ryan', 'condition' : 'synthetic_parafovea_04_EGOMOTION'}, 
            {'sequence' : 'groundtruth', 'subject' : 'ryan', 'condition' : 'synthetic_periphery_05_PRAIRIE'}, 
        }
    else:
        params['seed'] == 1 
        metadata = None

    return params, metadata, sequences, subjects, conditions

def run_analysis(params, metadata, sequences, subjects, conditions):
    """
    Carry out analysis.

    Prepare helper params objects. 03/28/24 will now use params dict directly.
    Classes will take in a params dict and internally instantiate an object of the
    ParamsDict class (validation that input is dict will occur in constructor
    method of ParamsDict).
    param_types = list(params.keys())
    params_obj = {type_ : ParamsDict(type_) for type_ in param_types}

    Input:
        params: Dictionary
    """

    # Create save directory iff it does not already exist. 
    save_dir = f'results/{params['preprocess']['experimenter']/{date.today()}}'
    subprocess.run(["mkdir", "-p", save_dir])


    for subj in subjects:
        for seq in sequences:
            for cond in conditions:
                experiment = (subj, seq, cond)

                # Load and preprocess data.
                data = PerceptualData(experiment, params['preprocess'])
                data.load_raw_data()
                data.preprocess()

                # Create synthetic null data.
                null_data = PerceptualDataNull(
                    experiment, 
                    params['preprocess'], 
                    params['curvature'],
                    params['null']
                )
                null_data.load_raw_data()
                null_data.preprocess()
                null_data.compute_pixel_curvature()
                null_data.synthesize()

                # Inference on perceptual data.
                inference = Inference(data, params['inference'])
                inference.infer_global_curvature()

                # Inference on synthetic null perceptual data.
                inference_null = Inference(null_data, params['inference'])
                inference_null.infer_curvature()

                # Bootstrapping on perceptual data.
                bootstrap = Bootstrapper(data, params['boot'])
                bootstrap.bootstrap()

                # Bootstrapping on perceptual data.
                bootstrap_null = Bootstrapper(null_data, params['boot'])
                bootstrap_null.bootstrap()

                # Collect results and save.
                results = {
                    'inference' : inference,
                    'inference_null' : inference_null,
                    'bootstrap' : bootstrap,
                    'bootstrap_null' : bootstrap_null
                }           
                torch.save(
                    results, 
                    f'{save_dir}/{subj}_{seq}_{params['general']['run_id']}'
                )


