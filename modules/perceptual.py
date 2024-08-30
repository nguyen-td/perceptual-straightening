import os
import torch
import warnings

from .curvature import PixelCurvature

class ExperimentalData():
    """
    Load and process raw experimental data. 
    """

    def __init__(self, experiment, params):
        """
        Class for loading and processing data for one experiment (defined by
        a subject, sequence, and condition).
        
        Arguments
        ---------
        experiment : 3-tuple, whose respective elements are:
            subject   : {str} Name of subject (e.g. 'alexandra').
            sequence  : {str} Name of sequence (e.g. 'natural_fovea_05_PRAIRIE').
            condition : {str} Name of condition (e.g., 'groundtruth')
        params     : dict whose key-value pairs store the relevant parameters 
                     for loading and processing data.
        """
        self.subject = experiment[0]        
        self.sequence = experiment[1]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
        self.condition = experiment[2]
        self.params = params
        self.raw = None
        # self.validate_params()

    def validate_params(self):
        # TODO
        pass

    def load_raw_data(self, load_dir=None):
        """
        Use path specified in params dict to attempt to load data.

        Arguments
        ---------

        """
        if 'load_dir' in self.params.keys():
            if load_dir is not None:
                warnings.warn(f"Overriding key word value for `load_dir` and using the value provided in the params dict: {self.params['load_dir']}.")
            load_dir = self.params['load_dir']

        data_dir = f'{load_dir}/{self.params['experimenter']}'
        # dir_file = f'{data_dir}/{self.sequence}.pt'
        dir_file = f'{data_dir}/{self.subject}_{self.condition}.pt'
        if os.path.isfile(dir_file): self.raw = torch.load(dir_file)
        
    def process(self, keep_raw=False):
        """
        Processes data to extract counts of correct and incorrect responses and
        compute proportion of correct response trials.

        Arguments
        ---------
        keep_raw : {Boolean | Default = False} Specify whether to keep raw data
                   as attribute (if True) or to delete it (if False). If 
                   'keep_raw' was provided as a key-value pair in params, it 
                   will override what is passed in here.
        """

        def corey_data(raw):
            """
            Special processing function for Corey's data. `raw` is a modified
            copy of `self.raw`. This is an encapsulation of an if-block in the
            original. TODO: write a separate routine to convert data to correct 
            format once and save result.
            """          
            raw[:, 1:2] = raw[:, 1:2] - 7
            raw[:, 3] = torch.zeros(raw.shape[0])

            for k in (1, 2):
                mask = (raw[:, k] <= 11) & (raw[:, k] >= 1)
                raw = raw[mask, k]

            return raw
      
        def count_correct_and_incorrect_responses(self):
            """
            Count correct and incorrect responses.

            Inputs:
            -------
            raw: (n_trials x n_columns) Torch array
                Array containing the raw data. The columns have the following meanings:
                    1. Natural (1) vs. synthetic (2)
                    2. A frame
                    3. B frame
                    4. Size, potentially of the stimulus
                    5. Unknown
                    6. What the X frame actually matches (e.g., 1 means it matches A, 2 means it matches B)
                    7. What the participant reported the X frame matches 
            """
            # Preallocate
            self.n_frames = torch.max(torch.cat((self.raw[:,1], self.raw[:,2]), 0)) 
            self.correct = torch.zeros(self.n_frames, self.n_frames)
            self.incorrect = torch.zeros(self.n_frames, self.n_frames)
            
            (cond, ind_i, ind_j, stim_size, crct) = (
                self.raw[:, 0], 
                self.raw[:, 1] - 1, # Indices are from MATLAB
                self.raw[:, 2] - 1, # Indices are from MATLAB
                self.raw[:, 3], 
                self.raw[:, 5] == self.raw[:, 6]
            )
            
            # Note that diagonals are empty because there are no trials where A = B, but they will be set to 0.5 below.
            for i_trial in range(self.raw.shape[0]):
                is_within_dev_range = (
                    self.params['min_deviation'] <= stim_size[i_trial] <= self.params['max_deviation']
                )
                has_positive_ind = ind_i[i_trial] > 0 and ind_j[i_trial] > 0

                if is_within_dev_range and has_positive_ind and crct[i_trial]:
                    self.correct[ind_i[i_trial].item(), ind_j[i_trial].item()] += 1
                else:
                    self.incorrect[ind_i[i_trial].item(), ind_j[i_trial].item()] += 1

        def complete_data(self):
            """
            Finish up processing.
            """
            # Compute total count and proportion of trials correct.
            self.total = self.correct + self.incorrect
            zero_trials = (self.total == 0).to(torch.float64)
            self.p_correct = self.correct / (self.total + zero_trials)

            # For each condition, set diagonal of n_frames x n_frames slice to 0.5.
            # TODO: Check to ensure this operation makes sense.
            torch.diagonal(self.p_correct, offset=0).fill_(0.5)

        # Will be set to None in load_raw_data if file path doesn't exist.
        if self.raw is None: 
            warnings.warn(
                f"processing requested for {self.subject}, {self.sequence}, {self.condition}, but no data were loaded."
                )
            return

        # process data.
        self.raw[:, 0] = torch.ones(self.raw.shape[0]) # TODO: Why? The first column is necessary for distinguishing between natural and synthetic
        if self.params['experimenter'] == 'corey': self.raw = corey_data(self.raw)
        count_correct_and_incorrect_responses(self)
        complete_data(self)

        # Unless user requested otherwise, overwrite `raw` to save memory.
        if not keep_raw: self.raw = None


class PerceptualDataNull(ExperimentalData):
    """
    """
    # TODO: figure out what to do with that
    # default_params = {
    #     **PerceptualData.default_params,
    #     **PixelCurvature.default_params,
    #     # TODO: add stuff here
    #     }

    # def __init__(self, experiment, pparams, cparams):
    #     PerceptualData.__init__(experiment, pparams)
    #     PixelCurvature.__init__(experiment[-2:], cparams)

    def __init__(self, experiment, pparams, cparams):
        super().__init__(experiment, pparams)
        self.experiment = experiment
        self.cparams = cparams

    def compute_pixel_curvature(self):
        pixel_curvature = PixelCurvature(self.experiment[-2:], self.cparams)
        self.load_sequence()
        self.compute_curvature_pointwise()

    def synthesize(self):
        pass 




