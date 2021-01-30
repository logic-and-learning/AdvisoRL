# Imports
import numpy as np
import tensorflow as tf
import random
from hrl.dqn_network import create_net, create_linear_regression, create_target_updates
from baselines.feature_proxy_hrl import FeatureProxy

class MetaController:
    def __init__(self, sess, policy_name, options, option2file, rm, use_rm, learning_params, num_features, num_states, show_print, epsilon=0.1):
        
        self.show_print = show_print
        self.options = options
        self.option2file = option2file
        self.epsilon = epsilon
        self.gamma = learning_params.gamma
        self.rm = rm 
        self.use_rm = use_rm
        self.tabular_case = learning_params.tabular_case

        # This proxy adds the machine state representation to the MDP state
        self.feature_proxy = FeatureProxy(num_features, num_states, self.tabular_case)
        self.num_actions  = len(options)
        self.num_features = self.feature_proxy.get_num_features()        
        
        # network parameters
        num_hidden_layers = 2                 # this has no effect on the tabular case
        num_neurons = 64                      # this has no effect on the tabular case
        self.target_network_update_freq = 100 # this has no effect on the tabular case
        if self.tabular_case:
            lr = 0.7
            buffer_size = 1
            self.batch_size = 1
            self.learning_starts = 0 
        else:
            lr = 1e-3 
            buffer_size = 50000
            self.batch_size =  32
            self.learning_starts = 100

        # create dqn network
        self.neuralnet = MCNet(sess, self.num_actions, self.num_features, policy_name, self.tabular_case, learning_params.use_double_dqn, lr, num_neurons, num_hidden_layers)

        # create experience replay buffer
        self.er_buffer = MCReplayBuffer(buffer_size)
        self.step = 0

        # preprocessing action masks (for pruning useless options)
        self.mask = {}
        for u in self.rm.get_states():
            a_mask = np.ones(self.num_actions, dtype=np.float)
            if use_rm and not self.rm.is_terminal_state(u):
                a_mask = np.zeros(self.num_actions, dtype=np.float)                
                # Options that would move the RM to another state is useful
                useful_options = self.rm.get_useful_transitions(u)
                # looking for an exact match
                for i in range(self.num_actions):
                    if _is_match(option2file[i].split("&"), useful_options, True):
                        a_mask[i] = 1
                # if no exact match is found, we relax this condition and use any option that might be useful
                if np.sum(a_mask) < 1:
                    a_mask = np.zeros(self.num_actions, dtype=np.float)                
                    for i in range(self.num_actions):
                        if _is_match(option2file[i].split("&"), useful_options, False):
                            a_mask[i] = 1
            self.mask[u] = a_mask

    def _get_mask(self, u):
        return self.mask[u]

    def finish_option(self, option_id, true_props):
        option = self.options[option_id]
        u0 = option.get_initial_state()
        return u0 != option.get_next_state(u0, true_props)

    def get_option(self, option_id):
        option = self.options[option_id]
        rm_id, rm_u = option_id, option.get_initial_state()
        return rm_id, rm_u

    def learn(self, s1, u1, a, r, s2, u2, done, steps):
        # adding this experience to the buffer
        s1 = self.feature_proxy.add_state_features(s1, u1)
        s2 = self.feature_proxy.add_state_features(s2, u2)
        self.er_buffer.add(s1, a, r, s2, self._get_mask(u2), 1.0 if done else 0.0, self.gamma**steps)

        if len(self.er_buffer) > self.learning_starts:
            if self.show_print: print("MC: Learning", self.step)
            # Learning
            s1, a, r, s2, s2_mask, done, gamma = self.er_buffer.sample(self.batch_size)
            self.neuralnet.learn(s1, a, r, s2, s2_mask, done, gamma)
            self.step += 1
        
            # Updating the target network
            if self.step%self.target_network_update_freq == 0:
                if self.show_print: print("MC: Update network", self.step)
                self.neuralnet.update_target_network()

    def get_action_epsilon_greedy(self, s, u):
        # Before learning starts, the agent behaves completely random
        if len(self.er_buffer) <= self.learning_starts or random.random() < self.epsilon:
            # we have to pick a random option such that its mask is 1.0
            #mask = self._get_mask(u)
            useful_options = [i for i in range(self.num_actions)]# if mask[i] > 0]
            return random.choice(useful_options)
        return self.get_best_action(s, u)

    def get_best_action(self, s, u):
        s = self.feature_proxy.add_state_features(s, u).reshape((1,108)) #changed self.num_features to 108
        action_id = self.neuralnet.get_best_action(s, self._get_mask(u))
        return int(action_id)

def _is_match(option, useful_options, find_perfect_match):
    """
    returns True if 'option' is between the useful_options
    """
    for useful_option in useful_options:
        if len(option) == sum([1 for o in option if o in useful_option]):
            if not find_perfect_match or len(set(useful_option)) == len(set(option)):
                return True
    return False


class MCNet:
    """
    This is the network used to train the meta controllers
    """
    def __init__(self, sess, num_actions, num_features, policy_name, tabular_case, use_double_dqn, lr, num_neurons, num_hidden_layers):
        self.sess = sess
        self.num_features = num_features
        self.num_actions = num_actions
        self.tabular_case = tabular_case
        self.policy_name = policy_name

        # Inputs to the network
        self.s1 = tf.placeholder(tf.float64, [None, 108]) #changed num_features to 108, here and self.s2
        self.s1_mask = tf.placeholder(tf.float64, [None, num_actions])
        self.a = tf.placeholder(tf.int32)
        self.r = tf.placeholder(tf.float64)
        self.s2 = tf.placeholder(tf.float64, [None, 108])
        self.s2_mask = tf.placeholder(tf.float64, [None, num_actions])
        self.done  = tf.placeholder(tf.float64)
        self.gamma = tf.placeholder(tf.float64) # gamma depends on the number of executed steps

        # Creating target and current networks
        with tf.variable_scope(self.policy_name): # helps to give different names to this variables for this network
            # Defining regular and target neural nets
            if self.tabular_case:
                with tf.variable_scope("q_network") as scope:
                    q_values, _ = create_linear_regression(self.s1, 108, num_actions)
                    scope.reuse_variables()
                    q_target, _ = create_linear_regression(self.s2, 108, num_actions)
            else:
                with tf.variable_scope("q_network") as scope:
                    q_values, q_values_weights = create_net(self.s1, num_features, num_actions, num_neurons, num_hidden_layers)
                    if use_double_dqn:
                        scope.reuse_variables()
                        q2_values, _ = create_net(self.s2, num_features, num_actions, num_neurons, num_hidden_layers)
                with tf.variable_scope("q_target"):
                    q_target, q_target_weights = create_net(self.s2, num_features, num_actions, num_neurons, num_hidden_layers)
                self.update_target = create_target_updates(q_values_weights, q_target_weights)

            # get action with max Q-value such that is 'valid' (the '10000000' is just a big constant to remove invalid actions)
            self.best_action = tf.argmax(q_values - 10000000*(1.0-self.s1_mask), 1)

            # Optimizing with respect to q_target
            action_mask = tf.one_hot(indices=self.a, depth=num_actions, dtype=tf.float64)
            q_current = tf.reduce_sum(tf.multiply(q_values, action_mask), 1)
            q_max = tf.reduce_max(q_target - 10000000*(1.0-self.s2_mask), axis=1) # I remove invalid actions by adding a big negative contants to them
            q_max = q_max * (1.0-self.done) # dead ends must have q_max equal to zero
            q_target_value = self.r + tf.multiply(self.gamma, q_max) 
            q_target_value = tf.stop_gradient(q_target_value)

            loss = 0.5 * tf.reduce_sum(tf.square(q_current - q_target_value))
            if self.tabular_case:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
            else:
                optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            self.train = optimizer.minimize(loss=loss)

            # Initializing the network values
            self.sess.run(tf.variables_initializer(self._get_network_variables()))
            self.update_target_network() #copying weights to target net


    def _get_network_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.policy_name)

    def learn(self, s1, a, r, s2, s2_mask, done, gamma):
        self.sess.run(self.train, {self.s1: s1, self.a: a, self.r: r, self.s2: s2, self.s2_mask: s2_mask, self.done: done, self.gamma: gamma})

    def get_best_action(self, s1, s1_mask):
        s1      = s1.reshape((1,108)) #self.num_features to 108
        s1_mask = s1_mask.reshape((1,self.num_actions))
        return self.sess.run(self.best_action, {self.s1: s1, self.s1_mask: s1_mask})

    def update_target_network(self):
        if not self.tabular_case:
            self.sess.run(self.update_target)
        


class MCReplayBuffer(object):
    """
    This is the experience replay buffer for training the meta controllers
    """
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, s1, a, r, s2, s2_mask, done, gamma):
        data = (s1, a, r, s2, s2_mask, done, gamma)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        S1, A, R, S2, S2_MASK, DONE, GAMMA = tuple([[] for _ in range(7)])
        for i in idxes:
            s1, a, r, s2, s2_mask, done, gamma = self._storage[i]
            S1.append(np.array(s1, copy=False))
            A.append(np.array(a, copy=False))
            R.append(r)
            S2.append(np.array(s2, copy=False))
            S2_MASK.append(np.array(s2_mask, copy=False))
            DONE.append(done)
            GAMMA.append(gamma) # -> gamma**n_steps
        return np.array(S1), np.array(A), np.array(R), np.array(S2), np.array(S2_MASK), np.array(DONE), np.array(GAMMA)

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)




