from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm import tqdm
import pandas as pd
from functools import partial
import copy

class KNNSampler():
    @classmethod
    def load(cls, path):
        return joblib.load(path)

    def save(self, path):
        joblib.dump(self, path)

    def __init__(self, **kwargs):
        self.sampler = NearestNeighbors(**kwargs)

    def fit(self, df, X_columns, y_columns):
        X = df[X_columns]
        self.y = df[y_columns]
        self.X_columns = X_columns
        self.sampler.fit(X)
        return self

    def sample(self, df, sampling_weights = None, n_draws = 30 ,replace = True, pandas_sampling_args = {}, **kneighbors_args):
        # accept data frames or different types of arrays
        if df.__class__ == pd.DataFrame:
            neighbors = self.sampler.kneighbors(df[self.X_columns] ,**kneighbors_args)
        else:
            neighbors = self.sampler.kneighbors(df ,**kneighbors_args)

        distances = neighbors[0]
        indexes = neighbors[1]
        samples = []
        for row ,distance in tqdm(list(zip(indexes ,distances))):
            sample_weights = self._handle_weights(distance, sampling_weights)
            sample = self.y.iloc[row.flatten()].sample(n = n_draws, replace = replace, weights = sample_weights, **pandas_sampling_args).values.flatten().tolist()
            samples.append(sample)
        return samples

    def _handle_weights(self, distances, sampling_weights):

        if not sampling_weights is None:
            if callable(sampling_weights):
                sampling_weights = sampling_weights(distances)
                sampling_weights = np.nan_to_num(sampling_weights, copy=True, nan=0.0, posinf=999, neginf=-999)
            else:
                pass
            # checks wieghts sum to avoid 'wieghts sum to zero' error in pandas sampler
            if sampling_weights.sum() == 0:
                sampling_weights = None

        return sampling_weights

class GroupwiseMixIn:

    def __init__(self, estimator, df, group_columns):
        '''
        init saves states of group instances, group columns and will create a dictinoary of estimators
        to create new estimators in the dict, you need to create a new instance of GroupwiseMixIn
        '''
        groupby_obj = df.groupby(group_columns)
        self.estimators = {grp: copy.deepcopy(
            estimator) for grp, _ in groupby_obj}
        self.group_columns = group_columns
        self.base_estimator = copy.deepcopy(estimator)
        return

    def __getitem__(self, item):
        '''
        used for retrieving estimators for each group
        '''
        try:
            return self.estimators[item]
        except KeyError:
            if type(item) == int:
                return (list(self.estimators)[item], self.estimators[list(self.estimators)[item]])
            else:
                raise

    def __getattr__(self,item):
        '''used for calling estimator methods for all groups'''
        return partial(self._apply,method = item)

    def _apply(self, df=None, group_data_proc=None, error_handler='warn',method = None, proc_args={}, **kwargs):
        '''
        group_data_proc is a function that processes the input df (for each group) and outputs a dictinoary containing the pieces
        of input extracted from the df. the keys of the dictinoary should be the inputs for the called method
        '''
        if not df is None:
            groupby_obj = df.groupby(self.group_columns)
            return_dict = {}
            for grp, df in tqdm(groupby_obj):
                try:
                    method_inputs = group_data_proc(df, **proc_args)
                    assert isinstance(method_inputs, dict)
                    return_dict[grp] = getattr(self[grp], method)(
                        **method_inputs, **kwargs)
                except Exception as exc:
                    if error_handler == 'coerce':
                        pass
                    elif error_handler == 'warn':
                        print(f'Error with group {grp}: {repr(exc)}')
                    else:
                        raise
        else:
            return_dict = {}
            for grp, obj in tqdm(self.estimators.items()):
                try:
                    return_dict[grp] = getattr(self[grp], method)(**proc_args, **kwargs)
                except Exception as exc:
                    if error_handler == 'coerce':
                        pass
                    elif error_handler == 'warn':
                        print(f'Error with group {grp}: {repr(exc)}')
                    else:
                        raise

        return return_dict

