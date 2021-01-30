import tensorflow as tf
import os.path, time
import numpy as np
from qrm.policy_bank import PolicyBank, Policy
from qrm.dqn_network import create_net, create_linear_regression, create_target_updates
#import pdb
class PolicyBankDQN(PolicyBank):
    def __init__(self, sess, num_actions, num_features, learning_params, reward_machines):
        self.use_double_dqn = learning_params.use_double_dqn
        self.use_priority = learning_params.prioritized_replay
        self.num_hidden_layers = learning_params.num_hidden_layers
        self.num_neurons = learning_params.num_neurons
        # Creating the network
        super().__init__(sess, num_actions, num_features, learning_params, reward_machines)

    def _create_policy(self, policy_name):
        policy = PolicyDQN(self.sess, policy_name, self.s1, self.a, self.s2, self.IS_weights, 
                        self.use_priority, self.num_features, self.num_actions, self.learning_params.gamma, 
                        self.learning_params.lr, self.learning_params.tabular_case, self.use_double_dqn,
                        self.num_hidden_layers, self.num_neurons)
        return policy

    def reconnect(self):
        # Redefining connections between the different DQN networks
        num_policies = self.get_number_of_policies()
        batch_size = tf.shape(self.next_policies)[0]
        
        # concatenating q_target of every policy
        Q_target_all = tf.concat([self.policies[i].get_q_target_value() for i in range(len(self.policies))], 1)

        # Indexing the right target next policy
        aux_range = tf.reshape(tf.range(batch_size),[-1,1])
        aux_ones = tf.ones([1, num_policies], tf.int32)
        delta = tf.matmul(aux_range * num_policies, aux_ones) 
        Q_target_index = tf.reshape(self.next_policies+delta, [-1])
        Q_target_flat = tf.reshape(Q_target_all, [-1])
        Q_target = tf.reshape(tf.gather(Q_target_flat, Q_target_index),[-1,num_policies]) 
        # Obs: Q_target is batch_size x num_policies tensor such that 
        #      Q_target[i,j] is the target Q-value for policy "j" in instance 'i'

        # Matching the loss to the right Q_target
        for i in range(1,num_policies): # recall that policy '0' is the constant policy
            p = self.policies[i]
            # Adding the critic trainer
            p.add_optimizer(self.rewards[:,i], Q_target[:,i])
            # Now that everythiong is set up, we initialize the weights
            p.initialize_variables()
        
        # Auxiliary variables to train all the critics, actors, and target networks
        self.train = []
        for i in range(1,num_policies):
            p = self.policies[i]
            if self.use_priority:
                self.train.append(p.td_error)
            self.train.append(p.train)
    
    def learn(self, s1, a, s2, rewards, next_policies, IS_weights, has_been):
        # Learning
        values = {self.s1: s1, self.a: a, self.s2: s2, self.rewards: rewards, self.next_policies: next_policies, self.IS_weights: IS_weights, self.has_been: has_been}
        res = self.sess.run(self.train, values)
        if self.use_priority:
            # Computing new priorities (sum of the absolute td-errors)
            td_errors = np.array([np.abs(td_error) for td_error in res if td_error is not None])
            #td_errors_mean = np.mean(td_errors, axis=0)
            # Now I actually think that the maximum td-error is more informative than the mean for assigning priorities
            td_errors_max = np.max(td_errors, axis=0) 
            return td_errors_max

    def get_best_action(self, rm_id, rm_u, s1, add_noise=False):
        policy = self._get_policy(0, rm_u)
        #pdb.set_trace()
        return self.sess.run(policy.get_best_action(), {self.s1: s1})[0]


class PolicyDQN(Policy):
    def __init__(self, sess, policy_name, s1, a, s2, IS_weights, use_priority, num_features, num_actions, gamma, lr, tabular_case, use_double_dqn, num_hidden_layers, num_neurons):
        super().__init__(sess, policy_name)
        self.tabular_case = tabular_case
        self.s1 = s1
        self.a  = a
        self.s2 = s2
        self.IS_weights = IS_weights
        self.gamma = gamma
        self.lr = lr
        self.use_priority = use_priority

        self._initializeModel(num_features, num_actions, use_double_dqn, num_hidden_layers, num_neurons)

    def _initializeModel(self, num_features, num_actions, use_double_dqn, num_hidden_layers, num_neurons):
        
        with tf.variable_scope(self.scope_name): # helps to give different names to this variables for this network
            # Defining regular and target neural nets
            if self.tabular_case:
                with tf.variable_scope("q_network") as scope:
                    q_values, _ = create_linear_regression(self.s1, num_features, num_actions)
                    scope.reuse_variables()
                    q_target, _ = create_linear_regression(self.s2, num_features, num_actions)
                update_target = None # q_values and q_target are the same in the tabular case
            else:
                with tf.variable_scope("q_network") as scope:
                    q_values, q_values_weights = create_net(self.s1, num_features, num_actions, num_neurons, num_hidden_layers)
                    if use_double_dqn:
                        scope.reuse_variables()
                        q2_values, _ = create_net(self.s2, num_features, num_actions, num_neurons, num_hidden_layers)
                with tf.variable_scope("q_target"):
                    q_target, q_target_weights = create_net(self.s2, num_features, num_actions, num_neurons, num_hidden_layers)
                update_target = create_target_updates(q_values_weights, q_target_weights)

            # Q_values -> get optimal actions
            best_action = tf.argmax(q_values, 1)

            # getting current value for q(s1,a)
            action_mask = tf.one_hot(indices=self.a, depth=num_actions, dtype=tf.float64)
            q_current = tf.reduce_sum(tf.multiply(q_values, action_mask), 1)
            
            # getting the target q-value for the best next action
            if use_double_dqn:
                # DDQN
                best_action_mask = tf.one_hot(indices=tf.argmax(q2_values, 1), depth=num_actions, dtype=tf.float64)
                q_target_value = tf.reshape(tf.reduce_sum(tf.multiply(q_target, best_action_mask), 1), [-1,1])
            else:
                # DQN
                q_target_value = tf.reshape(tf.reduce_max(q_target, axis=1), [-1,1])
            
            # It is important to stop the gradients so the target network is not updated by minimizing the td-error
            q_target_value = tf.stop_gradient(q_target_value)

        # Adding relevant networks to the state properties
        self.best_action = best_action
        self.q_current = q_current
        self.q_target_value = q_target_value
        self.update_target = update_target
                    

    def add_optimizer(self, reward, q_target):
        with tf.variable_scope(self.scope_name): # helps to give different names to this variables for this network
            # computing td-error 'r + gamma * max Q_t'
            self.td_error = self.q_current - (reward + self.gamma * q_target)

            # setting loss function
            if self.use_priority: 
                # prioritized experience replay
                huber_loss = 0.5 * tf.square(self.td_error) # without clipping
                loss = tf.reduce_mean(self.IS_weights * huber_loss) # weights fix bias in case of using priorities
            else:
                # standard experience replay
                loss = 0.5 * tf.reduce_sum(tf.square(self.td_error)) 
            
            if self.tabular_case:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            else:
                #optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
                optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.train = optimizer.minimize(loss=loss)


    def initialize_variables(self):
        # Initializing the network values
        self.sess.run(tf.variables_initializer(self._get_network_variables()))
        self.update_target_networks() #copying weights to target net


    def _get_network_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope_name)

    def update_target_networks(self):
        if not self.tabular_case:
            self.sess.run(self.update_target)

    def get_best_action(self):
        return self.best_action

    def get_q_target_value(self):
        return self.q_target_value

