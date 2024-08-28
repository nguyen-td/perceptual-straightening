from abc import ABC, abstractmethod
import numpy as np
import scipy
import re
import torch
import warnings

from utils import Params


class Curvature(Params):
    """
    General base class for computing "curvature" of a sequence of n-dimensional
    points.
    """

    # TODO: figure out what to do with this
    # default_params = {
    #     **Params.default_params,
    #     'norm_order' : 2
    # }

    def __init__(self, params, kwargs_lock=True):
        super().__init__(params=params, kwargs_lock=kwargs_lock)

    @abstractmethod
    def load_sequence(self, **kwargs):
        # Sequence should be loaded and assigned as `self.x` and should be an m
        # x n array, where m is the number of datapoints, and n is the
        # dimensionality of the data.
        pass

    def compute_speed(self, **kwargs):
        """
        Compute velocities of an n-dimensional trajectory. Operates on `self.x`, 
        which is an 2D tensor, where the first dimension 
        corresponds to datapoints and the remaining dimension is of length n, 
        corresponding to the data's n dimensions.
        """
        self._update_from_kwarg(kwargs)

        n_points = self.x.shape[0]
        disp = self.x[:n_points-1, :] - self.x[1:n_points, :]
        direction = torch.norm(disp, p=self.norm_order, dim=1)

    def compute_dist_curvature(self):
        pass

    def compute_curvature_pointwise(self):
        assert self.x is not None, "`self.x` has value `None`."

        n_points = self.x.shape[0]
        self.pointwise_curvature = torch.empty(n_points)

        # TODO: missing value, replace 0
        for i_point in range(n_points):
            self.pointwise_curvature[i_point] = 0
        
    def compute_curvature(self):
        self.compute_curvature_pointwise()
        self.curvature = torch.sum(self.pointwise_curvature)

    def compute_average_curvature(self):
        pass

    def compute_dist_curv_acc(self):
        pass


class PixelCurvature(Curvature):
    """
    Class for computing curvature of a given sequence of images in the pixel domain.

    NOTE: So far only parameter seems to be path, so will not inherit from Params or
    use ParamsDict for the time being.
    """
    def __init__(self, experiment, cparams):
        self.sequence = experiment[0]
        self.condition = experiment[1]
        self.pparams = cparams

    def load_sequence(self, load_dir=None):
        """
        Load in sequence.

        Arguments
        ---------
        load_dir : (str | Default = 'stimuli/stimuli'). Path to directory where 
                   stimuli files are stored. Note that if a path is provided as
                   a key value pair in cparams (see constructor method, 
                   Arguments), that will supersede the key word argument here.
        """
        # These are the recognized sequences.
        KNOWN_SEQUENCES = {'04_EGOMOTION' : 0, '05_PRARIE' : 1, '06_DAM' : 2}

        # Get identifiers from sequence string.
        seq_ids = re.split('_', self.sequence)
        seq = dict()
        seq['type'], seq['ecnt'] = seq_ids[0], seq_ids[1]
        seq['name'] = f'{seq_ids[2]}_{seq_ids[3]}'
        seq['idx'] = KNOWN_SEQUENCES.get(seq['name'], None)
        if seq['idx'] is None:
            raise KeyError(
                (f"Sequence name does not match any of the currently recognized sequences: {list(KNOWN_SEQUENCES.keys())}.")
            )

        # Check for value in cparams and use that if present.
        if 'load_dir' in self.cparams.keys(): 
            if load_dir is not None:
                warnings.warn(
                    "Overriding key word value for `load_dir` and using the " 
                    f"value provided in the cparams dict: {self.cparams.load_dir}."
                )
            load_dir = self.cparams.load_dir

        # Load in sequence and reshape.
        load_file = (
            f'{load_dir}/{self.experimenter}/{seq['type']}_movie_frames_post.mat' 
        )
        data_mat = scipy.io.loadmat(load_file)
        self.x = torch.from_numpy(
            data_mat[f'frames_{seq['ecnt']}'][:, :, :, seq['idx']]
        )
        self.x = torch.reshape(self.x, (self.x.shape[0], -1))
