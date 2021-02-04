# Iterative automata learning for improving reinforcement learning

This project studies how learning an automaton presenting the temporal logic of rewards might help the reinforcement learning process.
The automaton learning happens simultaneously to reinforcement learning.

<!-- Created by Zhe Xu, Bo Wu, Ivan Gavran, Daniel Neider, Yousef Ahmad and Ufuk Topcu.
Modified by Jean-Raphaël Gaglione. -->
This work is based on these two papers:
- Xu, Z., Gavran, I., Ahmad, Y., Majumdar, R., Neider, D., Topcu, U., & Wu, B. (2020).
  [Joint Inference of Reward Machines and Policies for Reinforcement Learning](https://ojs.aaai.org/index.php/ICAPS/article/view/6756).
  _Proceedings of the International Conference on Automated Planning and Scheduling, 30(1), 590-598_.
- Neider, D., Gaglione, J.R., Gavran, I., Topcu, U., Wu, B., Xu, Z. (2021).
  Advice-Guided Reinforcement Learning in a non-Markovian Environment.
  _In Proceedings of the Thirty-Fifth AAAI Conference on Artificial Intelligence (AAAI-21). To appear._

RL code modified from Rodrigo Toro Icarte's codes in https://bitbucket.org/RToroIcarte/qrm/src/master/.
- Toro Icarte, Rodrigo and Klassen, Toryn Q. and Valenzano, Richard and McIlraith, Sheila A. (2018).
  Using Reward Machines for High-Level Task Specification and Decomposition in Reinforcement Learning.
  _Proceedings of the 35th International Conference on Machine Learning (ICML), 2112-2121_

AdvisoRL algorithm is implemented as **JIRP+PYSAT**.


The current code can only be run in Ubuntu Linux and does not support mac or windows systems unless in a virtual Ubuntu environment.

## Dependencies

QRM requires Python3.5 with these libraries:
[numpy](http://www.numpy.org/),
[tensorflow](https://www.tensorflow.org/),
and (optionally) [pygame](https://www.pygame.org/news).
We use pygame to visualize the Water World environment, but it is not required by QRM or any of our baselines.

JIRP+PYSAT also requires the
[pysat](https://pysathq.github.io/)
library.


## Running examples

### running code

	python3 run1.py --help

To run our method and our two baselines, move to the *src* folder and execute *run1.py*. This code receives 6 parameters: The RL algorithm to use (which might be "qlearning" (QAS), "ddqn" (D-DQN), "hrl", or "jirp", the last of which is our method), the environment (which might be "office", "craft", or "traffic"), the map (which is integer 0), the number of independent trials to run per map, whether to show plots of learned automata, and which algorithm to use for reward machine learning ("SAT", "RPNI", etc). For instance, the following command runs JIRP (SAT method) one time over map 0 of the office environment, while showing the learned automata:

	python3 run1.py --algorithm="jirp" --world="office" --map=0 --num_times=1 --al_algorithm=SAT --show_plots=1

With JIRP+PYSAT, hints for the reward machine learning can be specified through `--hint=<hint>`. The internal format of a hint is any DFA, but hints specified through command line are restricted to sequences of transitions that must appear in the learned DFA.
The SAT algorithm used by pysat can be specified with `--sat_algorithm=rc2|fm|gc3`.


### change parameters

In order to change the task being performed, move to the corresponding folder from the *experiments* folder and change the task index specified in the ground truth file found in the *tests* folder. For example, in order to run task 2 from the office world, set the index (on line 2) between the square brackets as indicated in 'experiments/office/tests/ground_truth.txt' before running *run1.py*:

	["../experiments/office/reward_machines/t%d.txt"%i for i in [2]]  # tasks

### results

All results are saved in '/plotdata' in .csv format, to store all attained rewards for all independent runs, and in .txt format, to store the rewards averaged across all independent runs. For example, the following files would be saved for task 1 in the office world running using the JIRP SAT method:

	officeworld1jirpsat.csv
	avgreward_officeworld1jirpsat.txt

### ploting results

	python3 export_summary2.py --help

In order to plot the results, execute *export_summary.py* while in *src*. This code receives 3 parameters: the RL algorithm to use (which may be “qlearning” (QAS), "ddqn" (D-DQN), “hrl”, “jirpsat”, “jirprpni”, “jirppysat” or “jirppysat:hint:<hint>”), the environment (which may be “office”, “craft”, or “traffic”), and the task index (which may be 0,1,2,3, or 4 in the office & craft environments and may be 1 in the traffic environment, where 0 means the average rewards across all tasks). For example, in order to plot the results for task 1 in the office world running using the JIRP SAT method:

	python3 export_summary2.py --algorithm=jirpsat --world=office --task=1

Multiple algorithms can be plotted at the same time by specifying several `--algorithm=...` arguments or by separating algorithm names with commas.
To plot them on several plots use `--subplots`.

	python3 export_summary2.py --algorithms=jirpsat,jirprpni --world=office --task=1


NOTE: You can only get plots for algorithm-world-task combinations that have already been run. Otherwise, an error should be returned.
