# The craft environment

The 'map' folder contains 11 randomly generated maps. Each grid position might contain a resource, workstation, wall, or being empty. A brief explanation of each symbol follows:

- 'A' is the agent
- 'X' is a wall
- 'a' is a tree
- 'b' is a toolshed
- 'c' is a workbench
- 'd' is grass
- 'e' is a factory
- 'f' is iron
- 'g' is gold
- 'h' is gem

The 'reward_machines' folder contains 10 tasks for this environment. These tasks are based on the 10 tasks defined by [Andreas et al.](https://arxiv.org/abs/1611.01796) for the crafting environment. The 'tests' folder contains 11 testing scenarios. Each test is associated with one map and includes the path to the 10 tasks defined for the craft environment. It also includes the optimal number of steps needed to solve each task in the given map (we precomputed them using value iteration). We use the optimal number of steps to normalize the discounted rewards in our experiments.

The 'options' folder is only used by the Hierarchical RL baselines. It defines a set of sensible options to tackle the tasks defined for this domain.