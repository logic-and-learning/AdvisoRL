from worlds.game import GameParams
from worlds.office_world import OfficeWorldParams

class TesterOfficeWorld:
    def __init__(self, experiment, gamma, data = None):
        if data is None:
            # Reading the file
            self.experiment = experiment
            f = open(experiment)
            lines = [l.rstrip() for l in f]
            f.close()
            # setting the test attributes
            self.tasks = eval(lines[1])
            optimal_aux = eval(lines[2])

            # I compute the optimal reward
            self.optimal = {}
            for i in range(len(self.tasks)):
                self.optimal[self.tasks[i]] = gamma ** (float(optimal_aux[i]) - 1)
            self.optimal_steps = optimal_aux[0]
        else:
            self.experiment = data["experiment"]
            self.tasks   = data["tasks"]
            self.optimal = data["optimal"]

    def get_dictionary(self):
        d = {}
        d["experiment"] = self.experiment
        d["tasks"] = self.tasks
        d["optimal"] = self.optimal
        return d

    def get_reward_machine_files(self):
        return self.tasks

    def get_task_specifications(self):
        return self.tasks

    def get_task_params(self, task_specification):
        return GameParams("officeworld", OfficeWorldParams())

    def get_task_rm_file(self, task_specification):
        return task_specification
