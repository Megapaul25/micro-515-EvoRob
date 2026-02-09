
# Challenge 1: Evolving control with openAI gym

**Goal:** Evolve a neural network controller and compare with PPO.  
**How:** Use Multi-Layer Perceptrons (MLPs) on agents in a ‘gym environment’  

---

![https://static.observableusercontent.com/files/3c7bd896860dfaa77ddbdb37ab46164bb8bc031f15d0dd3e3c4c03332ab2e35b33b9defd09fe8716719436f60a605b5e504e77675e69af2f4770f4923d3e6ecd][image7]               ![][image8]     
**Introduction**  
In this exercise, we dive deeper into robot optimization with robot learning. We utilize so-called [Gym environments](https://gymnasium.farama.org/) (developed by OpenAI). Gym environments provide a ‘plug-and-play’ interface for dynamic simulated **World**s. Here an agent receives 'observations' from the gym interface (based on the underlying physics). These observations can be used to calculate ‘actions' (see the diagram below) that are sent back to the gym interface to compute the state for the next time-step. Actions represent motor commands for a specific agent (e.g. torques) while observations are often sensor values (joint angles, acceleration of torso, distance to ground, RGB images). We will evolve the weights of a neural network (NN) controller to run as fast as possible in the [half-cheetah environment](https://gymnasium.farama.org/environments/mujoco/half_cheetah/).  
Our **Evaluate Individual** now contains two systems: the agent and the environment. We focus on the ‘brain’ of the agent i.e. the NN controller. We use a simple Multi-Layer Perceptron architecture (MLP) and evolve the weights. At the end, we want to compare our evolved controller with a Reinforcement-Learning method called [PPO](https://arxiv.org/pdf/1707.06347).

**EA**: (CMA)ES from previous exercise  
**World**: Evolve controller for the half-cheetah gym-environment  
![][image9]

| Checkout to the challenge1 branch:  git checkout challenge1 |
| :---- |

*Q1.0* Look into the controller class

* What type of hidden neurons are used? What is the size of each MLP layer, and how does that relate to the sensors and actuators of the ‘agent’?   
* Which signals do the input neurons receive and what signals do they provide?   
  	*Hint:* look at the documentation of the [half-cheetah environment](https://gymnasium.farama.org/environments/mujoco/half_cheetah/).   
* Compute the number of parameters of our neural network with x sensory inputs, y action outputs?

Find out where in the code does the agent interact with the gym environment

* Given the default parameters, how many time steps does our agent interact with the simulated world.  
* How is fitness calculated in evaluate\_individual?  
* What does the reward contain? 

Q1.1

| Play around with your EA. Can you improve the performance by tuning the settings? population size initialisation mutation rate Min/max genotype values |
| :---- |

Q1.2 Decide on the settings of your EA

| Run a final EvoRob experiment |
| :---- |

* Can you obtain a maximum fitness above 1500

| Retest your best evolved controller with a different seed than the one used during evolution (check the env.reset() function). Do you obtain the maximum performance consistent with the maximum fitness found during evolution? |
| :---- |

Continuously random seeding during optimization is a method to obtain more robust controllers. See [https://gymnasium.farama.org/api/env/](https://gymnasium.farama.org/api/env/) and [https://gymnasium.farama.org/api/utils/](https://gymnasium.farama.org/api/utils/) for more information.

| Implement random seeding during evolution by changing the env.reset() function call |
| :---- |

* Retest the best again controller with a different seed, is the performance more consistent now?

Q*1.3* Video

| Make a video of the agent. Hint: read the gym documentation on [render mode](https://gymnasium.farama.org/api/env/#gymnasium.Env.render) to see how to obtain an rgb image.  |
| :---- |

Q1.4 We want to compare evolutionary algorithms to deep reinforcement learning (DRL) to find the best parameters of our neural network. A standard benchmark algorithm of DRL in robotics is Proximal Policy Optimization ([PPO](https://arxiv.org/pdf/1707.06347)), because of its high performance and its low sensitivity to hyperparameter choices. Stable-Baselines3 ([https://stable-baselines3.readthedocs.io/](https://stable-baselines3.readthedocs.io/)) provides a high quality implementation of this algorithm that conveniently interfaces with the OpenAI gym-interface in 3 lines of code:

env \= gym.make(ENV\_NAME)  
ppo \= PPO("MlpPolicy", env, device\=torch.device('cpu'))  
ppo.learn(total\_timesteps\= n\_total\_steps)

It is nice to know that when with our own custom gym-environments we could easily implement PPO in a similar way.

* How would you fairly compare the EA vs PPO? Consider total compute time and reward function to get a sound comparison.

| Decide on the number of learning steps for the PPO algorithm and train the policy.  |
| :---- |
