from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from tqdm import tqdm
import numpy as np
import joblib


class ForestSampler:
    '''
    ForestSampler is a data sampler and conditional distribution estimator based on forest algorithms.
    it follows a fit-sample (akin to sklearns fit-transform) paradgim.

    it's made to be userfriendly and accepts dataframes as inputs.

    the strategy is to fit forests and instead of returning the mean or median pointiwse estimate, it returns
    all the samples in the terminal nodes for each forest.
    '''

    @classmethod
    def load(cls, path):
        return joblib.load(path)

    def save(self, path):
        joblib.dump(self, path)

    def __init__(self, output_type='numerical', forest_estimator='rf', **estimator_args):
        assert output_type in ['categorical', 'numerical']
        assert forest_estimator in ['rf', 'et']

        if (output_type == 'categorical') & (forest_estimator == 'rf'):
            estimator = RandomForestClassifier(**estimator_args)
        elif (output_type == 'categorical') & (forest_estimator == 'et'):
            estimator = ExtraTreesRegressor(**estimator_args)
        elif (output_type == 'numerical') & (forest_estimator == 'rf'):
            estimator = RandomForestRegressor(**estimator_args)
        elif (output_type == 'numerical') & (forest_estimator == 'et'):
            estimator = ExtraTreesRegressor(**estimator_args)
        else:
            raise ValueError('')

        self.estimator = estimator
        return

    def _fit_scaler(self, X, scaler, init_args={}, fit_args={}):

        if not scaler is None:
            avalible_scalers = {'minmaxscaler': MinMaxScaler, 'standardscaler': StandardScaler,
                                'robustscaler': RobustScaler}
            if scaler.__class__ == str:
                scaler = avalible_scalers[scaler.lower()]()
            scaler.fit(X, **fit_args)
        else:
            # keep scaler as none
            pass
        return scaler

    def _transform_scaler(self, X, scaler, transform_args={}, ):

        if not scaler is None:
            return scaler.transform(X, **transform_args)
        else:
            return X

    def fit(self, data, X_columns, y_columns, scaler=None, scaler_init_args={}):
        assert not '_NODE' in y_columns

        X = data[X_columns]
        y = data[y_columns]

        # forest regressor may need scalling in output since mse is calculated and multioutput regression
        # may get biased towars dimensions with highest values
        scaler = self._fit_scaler(X=data[y_columns], scaler=scaler,
                                  init_args=scaler_init_args)
        # transform inputs
        y_transformed = self._transform_scaler(data[y_columns].values, scaler)

        # fit estimator
        self.estimator.fit(X, y_transformed)

        # create node to data mapper
        node_indexes = self.estimator.apply(X)
        tree_node_values_mapper_list = []
        for tree in range(node_indexes.shape[1]):
            tree_node_values_mapper_list.append(dict(
                y.assign(_NODE=node_indexes[:, tree]).groupby('_NODE').apply(
                    lambda x: [x[col].values.tolist() for col in x if not col == '_NODE'])))

        self.X_columns = X_columns
        self.y_columns = y_columns
        self.tree_node_values_mapper = dict(enumerate(tree_node_values_mapper_list))
        self.scaler = scaler
        return self

    def sample(self, data):
        node_indexes = self.estimator.apply(data[self.X_columns])
        drawn_samples = []
        for sample in tqdm([*range(node_indexes.shape[0])]):
            row_samples = [[] for _ in range(len(self.y_columns))]
            for tree in range(node_indexes.shape[1]):
                node = node_indexes[sample, tree]
                sampled_values = self.tree_node_values_mapper[tree][node]
                for dimension in range(len(sampled_values)):
                    row_samples[dimension] += sampled_values[dimension]
            drawn_samples.append(np.array(row_samples))
        return drawn_samples