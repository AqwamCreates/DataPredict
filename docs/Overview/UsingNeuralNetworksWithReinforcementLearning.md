# Requirements

* Knowledge on how to build neural networks, which can be found [here](UsingNeuralNetworksPart1.md).

# What Is Reinforcement Learning?

Reinforcement learning is a way for our models to learn on its own without the labels.

We can expect our models to perform poorly at the start of the training but they will gradually improve over time.

# Getting Started

Most of the reinforcement learning neural netwworks here uses reward values to train our models. There are three variants of neural networks that follow this rule:

* Deep Q-Learning / DQN

* Deep SARSA

* Deep Expected SARSA

All these models contains reinforce() function and have similar input parameters. We will focus on Deep Q-Learning, but we can also apply what you will learn to the other two as well.

# The Basics

## Environment Feature Vector

An environment feature vector is a vector containing all the information related to model's environment. It can contain as many information such as:

* Distance

* Health

* Speed

An example of environment feature vector will look like this:

```lua
local environmentFeatureVector = {

  {1, -32, 234, 12, -97} -- 1 is added at first column for bias, but it is optional.

}
```

## Reward Value

This is the value where we reward or punish the models. The properties of reward value is shown below:

* Positive value: Reward

* Negative Value: Punishment

* Large value: Large reward / punishment

* Small value: Small reward / punishment

It is recommended to set the reward that is within the range of -1 <= (reward * learning rate) <= 1.

## Action Labels

Action label is a label produced by the model. This label can be a part of decision-making classes or classification classes. For example:

* Decision-making classes: "Up", "Down", "Left", "Right", "Forward", "Backward"

* Classification classes: 1, 2, 3, 4, 5, 6

## Reinforce Function

Upon calling reinforce() function, it will return two values, but we are interested in the first one for this tutorial.

```lua
local DQN = DataPredict.Models.QLearningNeuralNetwork.new() -- Create a new model object.

DQN:createLayers({4, 3, 2}) -- Setting up our layers.

DQN:setClassesList({"Up", "Down"}) -- Setting up our classes.

local actionLabel = DQN:reinforce(environmentFeatureVector, rewardValue) -- Run the reinforce() function.
```

Each time we use reinforce() function with input parameters in it, it will train the neural network.

Ensure that both environment feature vector and reward value are from the same state.

## Experience Replay

Additionally, you can add experience replay to your model. All you have to do is to call the setExperienceReplay() function.

```lua
local DQN = DataPredict.Models.QLearningNeuralNetwork.new() -- Create a new model object.

local UniformExperienceReplay = DataPredict.ExperienceReplays.UniformExperienceReplay.new()

DQN:setExperienceReplay(UniformExperienceReplay) -- Placing our experience replay object here.
```

## Wrapping It All Up

In this tutorial, you have learnt the starting point of the reinforcement learning neural networks. 

These only cover the basics. You can find more information here:

* [Reinforcement Learning Neural Network Base Model](../API/Models/ReinforcementLearningNeuralNetworkBaseModel.md)

* [Deep Q-Learning](../API/Models/QLearningNeuralNetwork.md)

* [Deep SARSA](../API/Models/StateActionRewardStateActionNeuralNetwork.md)

* [Deep Expected SARSA](../API/Models/ExpectedStateActionRewardStateActionNeuralNetwork.md) 
