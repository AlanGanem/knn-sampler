from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np
from tqdm import tqdm
import pandas as pd
from functools import partial
import copy


class KNNSampler():
    '''
    KNNSampler is a data sampler and conditional distribution estimator based on data proximity.
    it follows a fit-sample (akin to sklearns fit-transform) paradgim.

    it's made to be userfriendly and accepts dataframes as inputs.

    the strategy is to fit the sampler based on some features and associate these features with user defined weights.
    Than, during the sampling process, the new data is going to be queried on historical data and return samples
    from trainning set according to a function of the distance of the inference features to the fitted data. this
    function returns the probability of sampling a point, based on the distance matrix of the queried point to the
    fitted data. This function can be defined by the user, otherwise it will assume a unform sampling probabilty
    distribution.
    '''
    @classmethod
    def load(cls, path):
        return joblib.load(path)

    def save(self, path):
        joblib.dump(self, path)

    def __init__(self, **kwargs):
        self.sampler = NearestNeighbors(**kwargs)

    def _fit_scaler(self, X, scaler, init_args = {}, fit_args = {}):

        if not scaler is None:
            avalible_scalers = {'minmaxscaler':MinMaxScaler,'standardscaler':StandardScaler, 'robustscaler':RobustScaler}
            if scaler.__class__ == str:
                scaler = avalible_scalers[scaler.lower()]()
            scaler.fit(X, **fit_args)
        else:
            #keep scaler as none
            pass
        return scaler

    def _transform_scaler(self, X, scaler, transform_args = {}):

        if not scaler is None:
            return scaler.transform(X, **transform_args)
        else:
            return X

    def fit(self, data, X_columns, y_columns, feature_weights = None,
            scaler = None, scaler_init_args = {},scaler_fit_args = {}):

        '''
        :param data: DataFrame of data to fit
        :param X_columns: features to be queried
        :param y_columns: values returned by queries
        :param feature_weights: user defined feature weights
        :param scaler: features scaler. Str of one of the sklearn scalers (MinMaxScaler, StandardScaler, RobustScaler)
        :param scaler_init_args: scaler initializing args
        :param scaler_fit_args: scaler fitting args
        :return: KNNSampler fitted instance
        '''
        #handle feature weights
        if feature_weights is None:
            feature_weights = {i:1 for i in X_columns}
        elif feature_weights.__class__ in [list,set,tuple]:
            feature_weights = {list(X_columns)[i]:feature_weights[i] for i in range(len(feature_weights))}
        elif feature_weights.__class__ == dict:
            if set(feature_weights) != set(X_columns):
                raise ValueError('"feature_weights" keys must match exactly "X_columns"')
            #order the dict according to X_columns
            feature_weights = {k:feature_weights[k] for k in X_columns}
        elif feature_weights.__class__ != dict:
            raise TypeError(f'feature_weights must be one of [dict,list,tuple,set], not {feature_weights.__class__}')

        # transform feature weights dict into an array in the correct order
        feature_weights_array = np.array([v for k, v in feature_weights.items()])

        # fit scaler
        scaler = self._fit_scaler(X=data[X_columns], scaler=scaler,
                                       init_args=scaler_init_args, fit_args=scaler_fit_args)
        #transform inputs
        X = self._transform_scaler(data[X_columns].values, scaler)
        #multiply X by feature weights
        X = X*feature_weights_array
        #fit sampler
        self.sampler.fit(X)

        #save states
        self.scaler = scaler
        self.feature_weights_array = feature_weights_array
        self.feature_weights = feature_weights
        self.y = data[y_columns]
        self.X_columns = X_columns

        return self

    def sample(self, data, sampling_weights = None, n_draws = 30 ,replace = True, pandas_sampling_args = {}, **kneighbors_args):
        '''
        Samples from historical data based on feature proximity

        :param data: query data
        :param sampling_weights: callable that recieves an array of distances and returns an array of sampling weights
        :param n_draws: number of sapmle draws
        :param replace: wether to sample with replacement or not
        :param pandas_sampling_args: kwargs for pandas.DataFrame.sample()
        :param kneighbors_args: kwargs for NearestNeighbors.kneighbors()
        :return: an array containing row(query data index) X columns (samples for each query) X channels (sampled features)
        '''
        #transform inputs
        # accept data frames or different types of arrays
        if data.__class__ == pd.DataFrame:
            #multiply feature weights
            X = data[self.X_columns].values
        else:
            # multiply feature weights
            X = data

        #scale inputs
        X = self._transform_scaler(X,self.scaler)
        #apply weights
        X = X*self.feature_weights_array
        #sample
        neighbors = self.sampler.kneighbors(X, **kneighbors_args)

        distances = neighbors[0]
        indexes = neighbors[1]
        samples = []
        for row ,distance in tqdm(list(zip(indexes ,distances))):
            sample_weights = self._handle_weights(distance, sampling_weights)
            sample = self.y.iloc[row.flatten()].sample(
                n = n_draws, replace = replace, weights = sample_weights, **pandas_sampling_args
            ).values.tolist()
            samples.append(sample)
        samples = np.array(samples)
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