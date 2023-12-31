from enum import IntEnum, Enum


class DecisionMakingPolicy(IntEnum):
    EPSILON_GREEDY_POLICY = 1
    NOISE_NETWORK_POLICY = 2
    RANDOM_POLICY = 3
