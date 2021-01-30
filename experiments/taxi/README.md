# The taxi environment

Inspired by [Taxi-v3 gym](https://gym.openai.com/envs/Taxi-v3/) ([source on GitHub](https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py)).

It differs from the gym in that way:
- the passenger always start at location `A`
- the passenger destination is determined:
    - **task 1**: location `B`
    - **task 2**: location `C`
    - **task 3**: location `D`

The `options` folder is only used by the Hierarchical RL baselines. It defines a set of sensible options to tackle the tasks defined for this domain.
