from baselines.dqn import DQN

class PolicyBank:
    """
    This class handles one policy per reward machines
    """
    def __init__(self, sess, num_actions, num_features, learning_params, curriculum, reward_machines):
        # Adding one policy per reward machine
        self.policies = []
        rm = reward_machines[0]
        num_states = len(rm.get_states())
        policy_name = "Reward_Machine_%d"%0
        # policy = DQN(sess, policy_name, learning_params, curriculum, num_features, num_states, num_actions)
        policy = DQN(sess, policy_name, learning_params, curriculum, num_features, num_states, num_actions)
        self.policies.append(policy)

    def learn(self, rm_id):
        self.policies[rm_id].learn()

    def add_experience(self, rm_id, s1, u1, a, r, s2, u2, done):
        self.policies[rm_id].add_experience(s1, u1, a, r, s2, u2, done)

    def get_step(self, rm_id):
        return self.policies[rm_id].get_step()

    def get_number_features(self, rm_id):
        return self.policies[rm_id].get_number_features()

    def get_best_action(self, rm_id, s1, u1):
        return self.policies[rm_id].get_best_action(s1, u1)[0]

    def add_step(self, rm_id):
        self.policies[rm_id].add_step()

    def update_target_network(self, rm_id):
        self.policies[rm_id].update_target_network()
