import numpy as np
import tensorflow as tf
from qrm.dqn_network import create_net, create_linear_regression, create_target_updates
from baselines.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.feature_proxy import FeatureProxy
from common.schedules import LinearSchedule

class DQN:
    """
    This baseline solves the problem using standard q-learning over the cross product 
    between the RM and the MDP
    """
    def __init__(self, sess, policy_name, learning_params, curriculum, num_features, num_states, num_actions):
        # initialize attributes
        self.sess = sess
        self.learning_params = learning_params
        self.use_double_dqn = learning_params.use_double_dqn
        self.use_priority = learning_params.prioritized_replay
        self.policy_name = policy_name
        self.tabular_case = learning_params.tabular_case
        # This proxy adds the machine state representation to the MDP state
        self.feature_proxy = FeatureProxy(num_features, num_states, self.tabular_case)
        self.num_actions  = num_actions
        self.num_features = self.feature_proxy.get_num_features()
        # create dqn network
        self._create_network(learning_params.lr, learning_params.gamma, learning_params.num_neurons, learning_params.num_hidden_layers)
        # create experience replay buffer
        if self.use_priority:
            self.replay_buffer = PrioritizedReplayBuffer(learning_params.buffer_size, alpha=learning_params.prioritized_replay_alpha)
            if learning_params.prioritized_replay_beta_iters is None:
                learning_params.prioritized_replay_beta_iters = curriculum.total_steps
            self.beta_schedule = LinearSchedule(learning_params.prioritized_replay_beta_iters, initial_p=learning_params.prioritized_replay_beta0, final_p=1.0)
        else:
            self.replay_buffer = ReplayBuffer(learning_params.buffer_size)
            self.beta_schedule = None
        # count of the number of environmental steps
        self.step = 0

    def _create_network(self, lr, gamma, num_neurons, num_hidden_layers):
        total_features = self.num_features
        total_actions = self.num_actions

        # Inputs to the network
        # self.s1_extended = tf.placeholder(tf.float64, [None, self.learning_params.memory_size*total_features+1])
        self.s1_extended = tf.placeholder(tf.float64, [None, self.learning_params.memory_size])
        # self.s1 = tf.placeholder(tf.float64, [None, 1])
        self.a = tf.placeholder(tf.int32)
        self.r = tf.placeholder(tf.float64)
        # self.s2 = tf.placeholder(tf.float64, [None, self.learning_params.memory_size])
        self.s2_extended = tf.placeholder(tf.float64, [None, self.learning_params.memory_size])
        # self.s2 = tf.placeholder(tf.float64, [None, 1])
        self.done = tf.placeholder(tf.float64)
        self.IS_weights = tf.placeholder(tf.float64) # Importance sampling weights for prioritized ER

        # Creating target and current networks
        with tf.variable_scope(self.policy_name): # helps to give different names to this variables for this network
            # Defining regular and target neural nets
            if self.tabular_case:
                with tf.variable_scope("q_network") as scope:
                    q_values, _ = create_linear_regression(self.s1_extended, self.s1_extended.shape.dims[1], total_actions)
                    scope.reuse_variables()
                    q_target, _ = create_linear_regression(self.s2_extended, self.s2_extended.shape.dims[1], total_actions)
            else:
                with tf.variable_scope("q_network") as scope:
                    # q_values, q_values_weights = create_net(self.s1, total_features, total_actions, num_neurons, num_hidden_layers)
                    q_values, q_values_weights = create_net(self.s1_extended, self.s1_extended.shape.dims[1], total_actions, num_neurons, num_hidden_layers)

                    if self.use_double_dqn:
                        scope.reuse_variables()
                        # q2_values, _ = create_net(self.s2, total_features, total_actions, num_neurons, num_hidden_layers)
                        q2_values, _ = create_net(self.s2_extended, self.s2_extended.shape.dims[1], total_actions, num_neurons, num_hidden_layers)

                with tf.variable_scope("q_target"):
                    # q_target, q_target_weights = create_net(self.s2, total_features, total_actions, num_neurons, num_hidden_layers)
                    q_target, q_target_weights = create_net(self.s2_extended, self.s2_extended.shape.dims[1], total_actions, num_neurons, num_hidden_layers)

                self.update_target = create_target_updates(q_values_weights, q_target_weights)

            # Q_values -> get optimal actions
            self.best_action = tf.argmax(q_values, 1)

            # Optimizing with respect to q_target
            action_mask = tf.one_hot(indices=self.a, depth=total_actions, dtype=tf.float64)
            q_current = tf.reduce_sum(tf.multiply(q_values, action_mask), 1)

            if self.use_double_dqn:
                # DDQN
                best_action_mask = tf.one_hot(indices=tf.argmax(q2_values, 1), depth=total_actions, dtype=tf.float64)
                q_max = tf.reduce_sum(tf.multiply(q_target, best_action_mask), 1)
            else:
                # DQN
                q_max = tf.reduce_max(q_target, axis=1)

            # Computing td-error and loss function
            q_max = q_max * (1.0-self.done) # dead ends must have q_max equal to zero
            q_target_value = self.r + gamma * q_max
            q_target_value = tf.stop_gradient(q_target_value)
            if self.use_priority: 
                # prioritized experience replay
                self.td_error = q_current - q_target_value
                huber_loss = 0.5 * tf.square(self.td_error) # without clipping
                loss = tf.reduce_mean(self.IS_weights * huber_loss) # weights fix bias in case of using priorities
            else:
                # standard experience replay
                loss = 0.5 * tf.reduce_sum(tf.square(q_current - q_target_value))
            
            # Defining the optimizer
            if self.tabular_case:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
            else:
                optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            self.train = optimizer.minimize(loss=loss)
            
        # Initializing the network values
        self.sess.run(tf.variables_initializer(self._get_network_variables()))
        self.update_target_network() #copying weights to target net

    def _train(self, s1_ext, a, r, s2_ext, done, IS_weights):

        # if np.ndim(s1_ext)==3:
        #     s1_ext = s1_ext[0]
        # elif np.ndim(s1_ext)==1:
        #     s1_ext = [s1_ext]
        #
        # if np.ndim(s2_ext)==3:
        #     s2_ext = s2_ext[0]
        # elif np.ndim(s2_ext)==1:
        #     s2_ext = [s2_ext]
        s1_ext = s1_ext.reshape(self.learning_params.batch_size,self.learning_params.memory_size)
        s2_ext = s2_ext.reshape(self.learning_params.batch_size,self.learning_params.memory_size)


        if self.use_priority: 
            _, td_errors = self.sess.run([self.train,self.td_error], {self.s1_extended: s1_ext, self.a: a, self.r: r, self.s2_extended: s2_ext, self.done: done, self.IS_weights: IS_weights})
        else:
            self.sess.run(self.train, {self.s1_extended: s1_ext, self.a: a, self.r: r, self.s2_extended: s2_ext, self.done: done})
            td_errors = None
        return td_errors

    def get_number_features(self):
        return self.num_features

    def learn(self):
        if self.use_priority:
            experience = self.replay_buffer.sample(self.learning_params.batch_size, beta=self.beta_schedule.value(self.get_step()))
            s1, a, r, s2, done, weights, batch_idxes = experience
        else:
            s1, a, r, s2, done = self.replay_buffer.sample(self.learning_params.batch_size)
            weights, batch_idxes = None, None
        # td_errors = self._train(s1, a, r, s2, done, weights) # returns the absolute td_error, used s1[0] next
        if np.ndim(s1)==2 and np.ndim(s2)==2:
            if s1.shape[0]!=32 or s1.shape[1]!=200 or s2.shape[0]!=32 or s2.shape[1]!=200:
                s1
        else:
            s1
        td_errors = self._train(s1, a, r, s2, done, weights) # returns the absolute td_error

        if self.use_priority:
            new_priorities = np.abs(td_errors) + self.learning_params.prioritized_replay_eps
            self.replay_buffer.update_priorities(batch_idxes, new_priorities)

    def add_experience(self, s1, u1, a, r, s2, u2, done):
        s1 = self.feature_proxy.add_state_features(s1, u1)
        s2 = self.feature_proxy.add_state_features(s2, u2)
        s1 = s1.reshape(200)
        s2 = s2.reshape(200)
        self.replay_buffer.add(s1, a, r, s2, done)

    def get_step(self):
        return self.step

    def add_step(self):
        self.step += 1

    def get_best_action(self, s1_ext, u1):

        # to_add = self.feature_proxy.add_state_features(s1_ext[0][self.num_features*self.learning_params.memory_size-self.num_features+1:self.num_features*self.learning_params.memory_size], u1).reshape((1,self.num_features))
        # to_add = self.feature_proxy.add_state_features(s1_ext[0][-1], u1).reshape((1,1))
        # if s1_ext.ndim>1:
        #     to_add = self.feature_proxy.add_state_features(s1_ext[0][-1], u1).reshape((1, 1))
        # else:
        #     to_add = self.feature_proxy.add_state_features(s1_ext[-1], u1).reshape((1,1))
        #
        #
        #
        # for item in range(self.num_features):
        #     np.delete(s1_ext,0)
        #
        # for item in to_add[0]:
        #     np.append(s1_ext,item)

        # s1_ext = self.feature_proxy.add_state_features(s1_ext[0], u1).reshape((1,self.num_features*self.learning_params.memory_size+1))
        if s1_ext.ndim>1:
            s1_ext = self.feature_proxy.add_state_features(s1_ext[0], u1).reshape((1,self.learning_params.memory_size))
        else:
            s1_ext = self.feature_proxy.add_state_features(s1_ext, u1).reshape((1, self.learning_params.memory_size))

        # s1 = s1_ext[0][-1]
        # s1 = s1.reshape((1, 1))

        # return self.sess.run(self.best_action, {self.s1_extended: s1_ext, self.s1: s1})

        return self.sess.run(self.best_action, {self.s1_extended: s1_ext})


    def update_target_network(self):
        if not self.tabular_case:
            self.sess.run(self.update_target)
        
    def _get_network_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.policy_name)

