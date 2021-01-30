#import pdb
class CurriculumLearner:
    """
    Decides when to stop one task and which to execute next
    In addition, it controls how many steps the agent has given so far
    """
    def __init__(self, tasks, num_steps = 100, min_steps = 1000, total_steps = 10000):
        """Parameters
        -------
        tasks: list of strings
            list with the path to the rms for each task
        num_steps: int
            max number of steps that the agent has to complete the task.
            if it does it, we consider a hit on its 'suceess rate' 
            (this emulates considering the average reward after running a rollout for 'num_steps')
        min_steps: int
            minimum number of training steps required to the agent before considering moving to another task
        total_steps: int
            total number of training steps that the agent has to learn all the tasks
        """
        self.num_steps = num_steps
        self.min_steps = min_steps
        self.total_steps = total_steps
        self.tasks = tasks

        assert len(self.tasks) == 1, "There must be only one task in the learning automaton setting"
    
    def get_tasks(self):
        return self.tasks

    def restart(self):
        self.current_step = 0
        self.current_task = -1

    def add_step(self):
        self.current_step += 1

    def get_current_step(self):
        return self.current_step

    def stop_learning(self):
        return self.total_steps <= self.current_step

    def get_next_task(self):
        self.last_restart = -1
        self.current_task = (self.current_task+1)%len(self.tasks)
        return self.get_current_task()
    
    def get_current_task(self):
        return self.tasks[self.current_task]

    def stop_task(self, step):
        return self.min_steps <= step

