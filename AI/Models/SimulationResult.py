import pandas as pd

from AI.Models.DQNAgentType import DQNAgentType
from AI.Models.DecisionMakingPolicy import DecisionMakingPolicy
from AI.Models.SimulatorProperties import SimulatorProperties


class SimulationResult:
    properties: SimulatorProperties
    path: str
    data: pd.DataFrame

    def agent_to_pretty_string(self):
        dqn_type = DQNAgentType(self.properties.agent.dqn_type).name
        episodes = self.properties.max_episodes
        gain = (self.data.loc[self.data.index[-1], 'Agent'] - 1.0)
        decision_making_policy = DecisionMakingPolicy(self.properties.agent.decision_making_policy)
        if decision_making_policy == DecisionMakingPolicy.RANDOM_POLICY:
            return 'Random Agent'

        return f"{episodes} {dqn_type} ({gain :+.2%})"  # 0.1254 -> +12.54%

    def market_to_pretty_string(self):
        gain = (self.data.loc[self.data.index[-1], 'Market'] - 1.0)
        return f'Market ({gain :+.2%})'
