# The office environment

This environment consists of one fix grid map (which is shown in the paper) that contains the following elements:

- 'a' represents the marked location 'A'
- 'b' represents the marked location 'B'
- 'c' represents the marked location 'C'
- 'd' represents the marked location 'D'
- 'e' represents a mailbox
- 'f' represents a coffee machine
- 'g' represents the office

The 'reward_machines' folder contains 4 tasks for this environment: 

- Task 1: deliver coffee to the office.
- Task 2: deliver mail to the office.
- Task 3: deliver mail and coffee to the office.
- Task 4: patrol locations A, B, C, and D.

The test file ('./tests/office.txt') includes the 4 tasks and the optimal number of steps needed to solve each task (we use those values to normalize the discounted rewards in our experiments). The 'options' folder contains the list of options to be used by our Hierarchical RL baselines when solving the office environment.