import tensorflow as tf
import os.path, time
import numpy as np
#import pdb

class PolicyBank:
    """
    This class includes a list of policies (a.k.a neural nets) for decomposing reward machines
    """
    def __init__(self, sess, num_actions, num_features, learning_params, reward_machines):
        self.sess = sess
        self.num_actions = num_actions
        self.num_features = num_features
        self.learning_params = learning_params
        # Decomposing reward machines: We learn one policy per state in a reward machine
        t_i = time.time()
        policies_to_add = self._decompose_reward_machines(reward_machines)
        print("Decomposing RMs is done! (in %0.2f minutes)"%((time.time()-t_i)/60))
        # Inputs to the network
        self.s1 = tf.placeholder(tf.float64, [None, num_features])
        self.rewards = tf.placeholder(tf.float64, [None, len(policies_to_add)])
        self.next_policies = tf.placeholder(tf.int32, [None, len(policies_to_add)])
        self.s2 = tf.placeholder(tf.float64, [None, num_features])
        self.a = tf.placeholder(tf.int32)
        self.has_been = tf.placeholder(tf.int32)
        self.IS_weights = tf.placeholder(tf.float64) # Importance sampling weights for prioritized ER
        # Actually adding the policies
        self._add_policies(policies_to_add)

    def _transfer_reward_machines(self, reward_machines):
        self.reward_machines = reward_machines
        # Some machine states might have equivalent Q-functions
        # In those cases, we learn only one policy for them
        policies_to_add = []
        self.state2policy = {}
        # We add one constant policy for every terminal state
        policies_to_add.append("constant") # terminal policy has id '0'
        # Associating policies to each machine state
        for i in range(len(reward_machines)):
            rm = reward_machines[0]
            for ui in range(len(rm.get_states())):
                if rm.is_terminal_state(ui):
                 # terminal states goes to the constant policy
                    self.state2policy[ui] = 0
                else:
                    # associating a policy for this reward machine state
                    policy_id = None
                    for uj in self.state2policy:
                        # checking if we already have a policy for an equivalent reward machine
                        if rm.is_this_machine_equivalent(ui, reward_machines[0], uj):
                            print("Match: reward machine %d from state %d is equivalent to reward machine %d from state %d"%(i,ui,0,uj))
                            policy_id = self.state2policy[uj]
                            break
                    if policy_id is None:
                        # creating a new policy for this node
                        policy_id = len(policies_to_add)
                        policies_to_add.append("machine" + "_state" + str(ui))
                    self.state2policy[ui] = policy_id
        return policies_to_add

    def _decompose_reward_machines(self, reward_machines):
        self.reward_machines = reward_machines
        # Some machine states might have equivalent Q-functions
        # In those cases, we learn only one policy for them
        policies_to_add = []
        self.state2policy = {}
        # We add one constant policy for every terminal state
        policies_to_add.append("constant") # terminal policy has id '0'
        # Associating policies to each machine state
        for i in range(len(reward_machines[0])):
            rm = reward_machines[0]
            for ui in range(len(rm.get_states())):
                if rm.is_terminal_state(ui):
                 # terminal states goes to the constant policy
                    self.state2policy[ui] = 0
                else:
                    # associating a policy for this reward machine state
                    policy_id = None
                    for uj in self.state2policy:
                        # checking if we already have a policy for an equivalent reward machine
                        if rm.is_this_machine_equivalent(ui, reward_machines[1], uj):
                            print("Match: reward machine %d from state %d is equivalent to reward machine %d from state %d"%(i,ui,0,uj))
                            policy_id = self.state2policy[uj]
                            break
                    if policy_id is None:
                        # creating a new policy for this node
                        policy_id = len(policies_to_add)
                        policies_to_add.append("machine" + "_state" + str(ui))
                    self.state2policy[ui] = policy_id
        return policies_to_add

    def _add_policies(self, policies_to_add):
        # creating individual networks per policy
        self.policies = []
        for p in policies_to_add:
            if p == "constant":
                self.policies.append(ConstantPolicy(0.0, self.s2, self.num_features))
            else:
                self.policies.append(self._create_policy(p))
        # connecting all the networks into one big net
        self.reconnect()

    def _get_policy(self, rm_id, rm_u):
        policy_id = self.state2policy[rm_u]
        return self.policies[policy_id]

    def get_number_of_policies(self):
        return len(self.policies)

    def update_target_network(self):
        for i in range(1,len(self.policies)):
            # recall that "self.policies[0]" is the constant policy
            self.policies[i].update_target_networks()

    def select_rewards(self, rewards):
        """
        reward format:
           [R0, ..., Rn] where Ri is the list of rewards gotten by each state on the reward machine 'i'
        returns a single vector with the corresponding rewards given to every policy
        """
        policy_rewards = np.zeros(len(self.policies),dtype=np.float64)
        done = set()
        for i in range(len(rewards)):
            for j in range(len(rewards[i])):
                pos = self.state2policy[(j)]
                if pos not in done:
                    policy_rewards[pos] = rewards[i][j]
                    done.add(pos)
                elif policy_rewards[pos] != rewards[i][j]:
                    print("Error! equivalent policies are receiving different rewards!")
                    print("(%d,%d) -> pos %d"%(i,j,pos))
                    print("reward discrepancy:",policy_rewards[pos],"vs",rewards[i])
                    print("state2policy", self.state2policy)
                    print("rewards", rewards)
                    exit()
        return policy_rewards

    def select_next_policies(self, next_states):
        """
        next_states format:
           [U0, ..., Un] where Ui is the list of next states for each state on the reward machine 'i'
        returns a single vector with the corresponding next policy per each policy
        """
        next_policies = np.zeros(len(self.policies),dtype=np.float64)
        done = set()
        for j in range(len(next_states[0])):
                u = self.state2policy[j]
                u_next = self.state2policy[next_states[0][j]]
                if u not in done:
                    next_policies[u] = u_next
                    done.add(u)
                elif next_policies[u] != u_next:
                    print("Error! equivalent policies have different next policy!")
                    print("(%d,%d) -> (%d,%d) "%(i,j,u,u_next))
                    print("policy discrepancy:",next_policies[u],"vs",u_next)
                    print("state2policy", self.state2policy)
                    print("next_states", next_states)
                    exit()
        return next_policies


    # def select_rewards(self, rewards):
    #     """
    #     reward format:
    #        [R0, ..., Rn] where Ri is the list of rewards gotten by each state on the reward machine 'i'
    #     returns a single vector with the corresponding rewards given to every policy
    #     """
    #     policy_rewards = np.zeros(len(self.policies),dtype=np.float64)
    #     done = set()
    #     for i in range(len(rewards)):
    #         for j in range(len(rewards[i])):
    #             pos = self.state2policy[(i,j)]
    #             if pos not in done:
    #                 policy_rewards[pos] = rewards[i][j]
    #                 done.add(pos)
    #             elif policy_rewards[pos] != rewards[i][j]:
    #                 print("Error! equivalent policies are receiving different rewards!")
    #                 print("(%d,%d) -> pos %d"%(i,j,pos))
    #                 print("reward discrepancy:",policy_rewards[pos],"vs",rewards[i][j])
    #                 print("state2policy", self.state2policy)
    #                 print("rewards", rewards)
    #                 exit()
    #     return policy_rewards

    # def select_next_policies(self, next_states):
    #     """
    #     next_states format:
    #        [U0, ..., Un] where Ui is the list of next states for each state on the reward machine 'i'
    #     returns a single vector with the corresponding next policy per each policy
    #     """
    #     next_policies = np.zeros(len(self.policies),dtype=np.float64)
    #     done = set()
    #     for i in range(len(next_states)):
    #         for j in range(len(next_states[i])):
    #             u = self.state2policy[(i,j)]
    #             u_next = self.state2policy[(i,next_states[i][j])]
    #             if u not in done:
    #                 next_policies[u] = u_next
    #                 done.add(u)
    #             elif next_policies[u] != u_next:
    #                 print("Error! equivalent policies have different next policy!")
    #                 print("(%d,%d) -> (%d,%d) "%(i,j,u,u_next))
    #                 print("policy discrepancy:",next_policies[u],"vs",u_next)
    #                 print("state2policy", self.state2policy)
    #                 print("next_states", next_states)
    #                 exit()
    #     return next_policies

    # To implement...
    def _create_policy(self, policy_name):
        raise NotImplementedError("To be implemented")

    def reconnect(self):
        raise NotImplementedError("To be implemented")

    def learn(self, s1, a, s2, rewards, next_policies, IS_weights):
        raise NotImplementedError("To be implemented")

    def get_best_action(self, rm_id, rm_u, s1, add_noise=False):
        raise NotImplementedError("To be implemented")


class ConstantPolicy:
    def __init__(self, value, s2, num_features):
        self._initialize_model(value, s2, num_features)

    def _initialize_model(self, value, s2, num_features):
        W = tf.constant(0, shape=[num_features, 1], dtype=tf.float64)
        b = tf.constant(value, shape=[1], dtype=tf.float64)
        self.q_target_value = tf.matmul(s2, W) + b

    def get_q_target_value(self):
        # Returns a vector of 'value' 
        return self.q_target_value

class Policy:
    def __init__(self, sess, policy_name):
        self.sess = sess
        self.scope_name = policy_name

    def update_target_networks(self):
        raise NotImplementedError("To be implemented")

