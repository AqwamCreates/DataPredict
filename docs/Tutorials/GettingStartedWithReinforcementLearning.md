# Requirements

* Knowledge on how to build neural networks, which can be found [here](UsingNeuralNetworksPart1.md).

# What Is Reinforcement Learning?

Reinforcement learning is a way for our models to learn on its own without the labels.

We can expect our models to perform poorly at the start of the training but they will gradually improve over time.

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

It is recommended to set the reward that is within the range of:

```lua
-1 <= (total reward * learning rate) <= 1
```

## Action Labels

Action label is a label produced by the model. This label can be a part of decision-making classes or classification classes. For example:

* Decision-making classes: "Up", "Down", "Left", "Right", "Forward", "Backward"

* Classification classes: 1, 2, 3, 4, 5, 6

# Setting Up Our Reinforcement Learning Model

There are two ways we can setup our models for our environment:

* Classical Setup (Recommended for experts or people who wants more control over their models)

* Quick Setup (Recommended for beginners)

Below we will show you the difference between the two above

## Classical Setup


## Quick Setup

# Wrapping It All Up

In this tutorial, you have learnt the starting point of the reinforcement learning neural networks. 

These only cover the basics. You can find more information here:

* [Reinforcement Learning Neural Network Base Model](../API/Models/ReinforcementLearningNeuralNetworkBaseModel.md)

* [Deep Q-Learning](../API/Models/QLearningNeuralNetwork.md)

* [Deep SARSA](../API/Models/StateActionRewardStateActionNeuralNetwork.md)

* [Deep Expected SARSA](../API/Models/ExpectedStateActionRewardStateActionNeuralNetwork.md) 
