import numpy as np

class FeatureProxy:
    def __init__(self, num_features, num_states, is_tabular):
        self.num_features = num_features
        self.num_states   = num_states
        self.is_tabular = is_tabular

    def get_num_features(self):
        if self.is_tabular:
            return self.num_states * self.num_features
        else:
            return self.num_states + self.num_features

    def add_state_features(self, s, u_i):
        u_i = 0
        if self.is_tabular:
            ret = np.zeros((1, self.num_features)) #self.num_states changed to 1
            ret[u_i,:] = s
            ret = ret.ravel() # to 1D
        else:
            ret = np.concatenate((s,self._get_one_hot_vector(u_i))) # adding the DFA state to the features
        return ret

    def _get_one_hot_vector(self, u_i):
        one_hot = np.zeros((self.num_states), dtype=np.float64)
        one_hot[u_i] = 1.0
        return one_hot