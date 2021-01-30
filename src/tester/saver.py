import os
from tester.test_utils import save_json

class Saver:
    def __init__(self, alg_name, tester, curriculum):
        folder = "../tmp/"

        # e.g. tester.experiment = "../experiments/craft/tests/craft_0.txt"
        exp_name = tester.experiment.split("/")
        exp_dir = os.path.join(folder, exp_name[2], exp_name[4].replace(".txt",""))
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        self.file_out = os.path.join(exp_dir, alg_name + ".json")
        self.tester = tester

    def save_results(self):
        results = {}
        results['game_type'] = self.tester.game_type
        results['steps'] = self.tester.steps      
        results['results'] = self.tester.results
        results['world'] = self.tester.get_world_dictionary()

        save_json(self.file_out, results)
