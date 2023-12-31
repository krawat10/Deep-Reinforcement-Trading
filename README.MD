<br />
<div align="center">
  <h3 align="center">Trading Bot Algorithm</h3>

  <p align="center">
    Deep Reinforcement Learning Trading Bot that uses Double Deep Q-Learning Algorithm + Description in Master Thesis 
  </p>
</div>

## Requirements

* Python 3.9.7 :: Anaconda, Inc. on win32
* pip 22.1.1
## Results
  
<div align="center">

![ddqn alghoritm](/ddqn_example_with_fresh_data.png?raw=true "Double Deep Q-Learning Bot Performance")

Fig. 1. DDQN Agent performance (top chart) compared to market (bottom chart). 
The agent was able to save the asset value by short positions or holding cash. 
The following chart shows that the agent can be a good hedge against market losses. 
🟩 `Green lines - periods of time where the agent keeps a long position,` 
🟦 `blue lines - periods of time where the agent keeps cash,` 
🟥 `red lines - periods of time where the agent keeps a short position.` The data used in the following chart wasn’t used to train the network (fresh data).

<br> 

![random alghoritm](/random_example_with_fresh_data.png?raw=true "Random Bot Performance")

Fig. 2. Random agent performance (top chart) compared to market (bottom chart). 

</div>
