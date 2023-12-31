import json

from AI.Models.DecisionMakingPolicy import DecisionMakingPolicy
from AI.Models.DQNAgentType import DQNAgentType


class DQNProperties:
    # Define network input and output
    state_dim: int
    num_actions: int

    state_names = []

    # Hyper parameters
    gamma = 0.99
    tau = 100

    # greedy Policy
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay_steps = 250
    epsilon_exponential_decay = 0.99
    decision_making_policy = DecisionMakingPolicy.EPSILON_GREEDY_POLICY
    dqn_type: DQNAgentType = DQNAgentType.DDQN

    # Experience Replay
    replay_capacity = int(1e6)
    batch_size = 4096

    # NN Architecture
    l2_reg = 1e-6
    learning_rate = 0.0001
    architecture = (1024, 1024)

    saved_network_dir = None

    def toJSON(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__)

    @staticmethod
    def from_dictionary(dic):
        dqn_properties = DQNProperties()

        # Update attributes with values from JSON data
        dqn_properties.state_dim = dic.get('state_dim')
        dqn_properties.num_actions = dic.get('num_actions')
        dqn_properties.gamma = dic.get('gamma', 0.99)  # Default to 0.99 if 'gamma' is not present
        dqn_properties.tau = dic.get('tau', 100)  # Default to 100 if 'tau' is not present
        dqn_properties.epsilon_start = dic.get('epsilon_start', 1.0)
        dqn_properties.epsilon_end = dic.get('epsilon_end', 0.01)
        dqn_properties.epsilon_decay_steps = dic.get('epsilon_decay_steps', 250)
        dqn_properties.epsilon_exponential_decay = dic.get('epsilon_exponential_decay', 0.99)
        dqn_properties.decision_making_policy = dic.get('decision_making_policy', DecisionMakingPolicy.EPSILON_GREEDY_POLICY)
        dqn_properties.dqn_type = dic.get('dqn_type', DQNAgentType.DDQN)
        dqn_properties.replay_capacity = dic.get('replay_capacity', int(1e6))
        dqn_properties.batch_size = dic.get('batch_size', 4096)
        dqn_properties.l2_reg = dic.get('l2_reg', 1e-6)
        dqn_properties.learning_rate = dic.get('learning_rate', 0.0001)
        dqn_properties.architecture = dic.get('architecture', (1024, 1024))
        dqn_properties.saved_network_dir = dic.get('saved_network_dir')

        return dqn_properties
