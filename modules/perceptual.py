import os
import torch
import warnings

from .curvature import PixelCurvature
from utils.params import Params


class PerceptualData(Params):
    """
    """
    # TODO: figure out what this is used for
    # default_params = {
    #     **Params.default_values,
    #     'min_dev' : float('-inf'),
    #     'max_dev' : float('inf')
    #     }

    def __init__(self, experiment, params, kwargs_lock=True):
        """
        Class for loading and preprocessing data for one experiment (defined by
        a subject, sequence, and condition).
        
        Arguments
        ---------
        experiment : 3-tuple, whose respective elements are:
            subject   : {str} Name of subject (e.g. 'alexandra').
            sequence  : {str} Name of sequence (e.g. 'natural_fovea_05_PRAIRIE').
            condition : {str} Name of condition (e.g., 'groundtruth')
        params     : dict whose key-value pairs store the relevant parameters 
                     for loading and preprocessing data.
        """
        self.subject = experiment[0]        
        self.sequence = experiment[1]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
        self.condition = experiment[2]
        self.raw = None
        self.kwargs_lock = kwargs_lock
        self._set_params(params)
        self.validate_params()

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
                warnings.warn(
                    "Overriding key word value for `load_dir` and using the " 
                    f"value provided in the params dict: {self.params.load_dir}."
                )
            load_dir = self.params.load_dir

        data_dir = f'{load_dir}/{self.params.experimenter}'
        dir_file = f'{data_dir}/{self.sequence}.pt'
        if os.path.isfile(dir_file): self.raw = torch.load(data_dir)
        
    def preprocess(self, keep_raw=False):
        """
        Preprocesses data to extract counts of correct and incorrect responses and
        compute proportion of correct response trials.

        Arguments
        ---------
        keep_raw : {Boolean | Default = False} Specify whether to keep raw data
                   as attribute (if True) or to delete it (if False). If 
                   'keep_raw' was provided as a key-value pair in params, it 
                   will override what is passed in here.
        """
        # ---------------------------------------------------------------
        def corey_data(raw):
            """
            Special preprocessing function for Corey's data. `raw` is a modified
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
      
        def count_correct_and_incorrect_responses(self, raw):
            """
            TODO: Something more descriptive here.
            """
            # Preallocate. TODO: get rid of n_conds.
            n_conds, self.n_frames = torch.max(raw[:, 0]), torch.raw(max[:, 1])
            self.correct, self.incorrect = torch.full(
                (n_conds, self.n_frames, self.n_frames), 
                torch.nan
            )
            # self.correct, self.incorrect = torch.zeros(
            #     n_conds, self.n_frames, self.n_frames
            # )
            
            (cond, ind_i, ind_j, devi, crct) = (
                raw[:, 0], 
                raw[:, 1] - 1, # Indices are from MATLAB
                raw[:, 2] - 1, # Indices are from MATLAB
                raw[:, 3], 
                raw[:, 5] == raw[:, 6]
            )
            
            # XXX: Olivier wrote here for t = 1, raw:size(2), which seems to
            # entail iteration over columns rather than rows of raw. But since
            # it is the rows that seem to correspond to data points, it may be
            # for t = 1, raw:size(1) is what is needed. Need to double check
            # interpretation of data structure. Do rows correspond to trials???
            # Yes. Also, note that diagonals are empty because there are no
            # trials where A = B, but they will be set to 0.5 below.
            for i_trial in range(raw.shape[0]):
                is_within_dev_range = (
                    self.params.min_dev < devi[i_trial] < self.params.max_dev
                )
                has_positive_ind = ind_i[i_trial] > 0 and ind_j[i_trial] > 0

                if is_within_dev_range and has_positive_ind and crct[i_trial]:
                    self.correct[cond[i_trial]][ind_i[i_trial]][ind_j[i_trial]] += 1
                else:
                    self.incorrect[cond[i_trial]][ind_i[i_trial]][ind_j[i_trial]] += 1

                # TODO: Check if any off diagonal elements are still nan and
                # warn if so.

        def complete_data(self):
            """
            Finish up preprocessing.
            """
            # Compute total count (TODO: of trials?) and proportion of trials correct.
            self.all = torch.clone(self.correct) + self.incorrect
            zero_trials = (self.all == 0).to(torch.float64)
            self.p_correct = self.correct / (self.all + zero_trials)

            # For each condition, set diagonal of n_frames x n_frames slice to 0.5.
            # TODO: Check to ensure this operation makes sense.
            torch.diagonal(self.p_correct, offset=0, dim1=-2, dim2=-1).fill_(0.5)
        # ---------------------------------------------------------------

        # Will be set to None in load_raw_data if file path doesn't exist.
        if self.raw is None: 
            warnings.warn(
                f"Preprocessing requested for {self.subject}, {self.sequence}, {self.condition}, but no data were loaded."
                )
            return

        # Preprocess data.
        raw = torch.clone(self.raw)
        raw[:, 0] = torch.ones(raw.shape[0])
        if self.params.experimenter == 'corey': raw = corey_data(raw)
        count_correct_and_incorrect_responses(self, raw)
        complete_data(self)

        # Unless user requested otherwise, overwrite `raw` to save memory.
        if 'keep_raw' in self.params.keys(): keep_raw = self.params.keep_raw
        if not keep_raw: self.raw = None


class PerceptualDataNull(PerceptualData, PixelCurvature):
    """
    """
    # TODO: figure out what to do with that
    # default_params = {
    #     **PerceptualData.default_params,
    #     **PixelCurvature.default_params,
    #     # TODO: add stuff here
    #     }

    def __init__(self, experiment, pparams, cparams, nparams, kwargs_lock=True):
        PerceptualData.__init__(experiment, pparams)
        PixelCurvature.__init__(experiment[-2:], cparams)
        Params.__init__(nparams, kwargs_lock=kwargs_lock)

    def compute_pixel_curvature(self):
        self.load_sequence()
        self.compute_curvature_pointwise()

    def synthesize(self):
        pass 




