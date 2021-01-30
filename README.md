# Iterative automata learning for improving reinforcement learning

This project studies how learning an automaton presenting the temporal logic of rewards might help the reinforcement learning process.
The automaton learning happens simultaneously to reinforcement learning.

Created by Zhe Xu, Bo Wu, Ivan Gavran, Daniel Neider, Yousef Ahmad and Ufuk Topcu.

RL code modified from Rodrigo Toro Icarte's codes in https://bitbucket.org/RToroIcarte/qrm/src/master/.

The current code can only be run in Ubuntu Linux and does not support mac or windows systems unless in a virtual Ubuntu environment.


## Running examples

To run our method and our two baselines, move to the *src* folder and execute *run1.py*. This code receives 6 parameters: The RL algorithm to use (which might be "qlearning" (QAS), "ddqn" (D-DQN), "hrl", or "jirp", the last of which is our method), the environment (which might be "office", "craft", or "traffic"), the map (which is integer 0), the number of independent trials to run per map, whether to show plots of learned automata, and whether to use the SAT algorithm or the RPNI algorithm. For instance, the following command runs JIRP (SAT method) one time over map 0 of the office environment, while showing the learned automata:

	python3 run1.py --algorithm="jirp" --world="office" --map=0 --num_times=1 --is_SAT=1 --show_plots=1

In order to change the task being performed, move to the corresponding folder from the *experiments* folder and change the task index specified in the ground truth file found in the *tests* folder. For example, in order to run task 2 from the office world, set the index (on line 2) between the square brackets as indicated in 'experiments/office/tests/ground_truth.txt' before running *run1.py*:

	["../experiments/office/reward_machines/t%d.txt"%i for i in [2]]  # tasks

All results are saved in '/plotdata' in .csv format, to store all attained rewards for all independent runs, and in .txt format, to store the rewards averaged across all independent runs. For example, the following files would be saved for task 1 in the office world running using the JIRP SAT method:

	officeworld1jirpsat.csv
	avgreward_officeworld1jirpsat.txt

In order to plot the results, execute *export_summary.py* while in *src*. This code receives 3 parameters: the RL algorithm to use (which may be “qlearning” (QAS), "ddqn" (D-DQN), “hrl”, “jirpsat”, or “jirprpni”), the environment (which may be “office”, “craft”, or “traffic”), and the task index (which may be 0,1,2,3, or 4 in the office & craft environments and may be 1 in the traffic environment, where 0 means the average rewards across all tasks). For example, in order to plot the results for task 1 in the office world running using the JIRP SAT method:

	python3 export_summary.py --algorithm=jirpsat --world=office --task=1

NOTE: You can only get plots for algorithm-world-task combinations that have already been run. Otherwise, an error should be returned.




