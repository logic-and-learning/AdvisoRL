from worlds.traffic_world import TrafficWorldParams, TrafficWorld
from worlds.craft_world import CraftWorldParams, CraftWorld
from worlds.office_world import OfficeWorldParams, OfficeWorld

class GameParams:
    """
    Auxiliary class with the configuration parameters that the Game class needs
    """
    def __init__(self, game_type, game_params):
        self.game_type   = game_type
        self.game_params = game_params
        if self.game_type not in ["craftworld", "trafficworld", "officeworld"]:
            print(self.game_type, "is not currently supported")
            exit()

class Game:

    def __init__(self, params):
        self.params = params
        if params.game_type == "craftworld":
            self.game = CraftWorld(params.game_params)
        if params.game_type == "trafficworld":
            self.game = TrafficWorld(params.game_params)
        if params.game_type == "officeworld":
            self.game = OfficeWorld(params.game_params)
        
    def is_env_game_over(self):
        return self.game.env_game_over

    def execute_action(self, action):
        """
        We execute 'action' in the game
        Returns the reward
        """
        self.game.execute_action(action)

    def get_actions(self):
        """
        Returns the list with the actions that the agent can perform
        """
        return self.game.get_actions()

    def get_last_action(self):
        """
        Returns agent's last performed action
        """
        return self.game.get_last_action()

    def get_true_propositions(self):
        """
        Returns the string with the propositions that are True in this state
        """
        return self.game.get_true_propositions()

    def get_true_propositions_action(self,a):
        """
        Returns the string with the propositions that are True in this state
        """
        return self.game.get_true_propositions_action(a)

    def get_state(self):
        """
        Returns a representation of the current state with enough information to 
        compute a reward function using an RM (the format is domain specific)
        """
        return self.game.get_state()

    # The following methods return different feature representations of the map ------------
    def get_features(self):
        return self.game.get_features()
    
    def get_state_and_features(self):
        return self.get_state(), self.get_features()
