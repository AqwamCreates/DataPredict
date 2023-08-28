# Requirements

* Knowledge on how to build neural networks, which can be found [here](UsingNeuralNetwork.md).

$ What Is Reinforcement Learning?

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

```
local environmentFeatureVector = {{1, 4345, 234, 1234}} -- 1 is added at first column for bias, but it is not neccessary.
```

### Reward Value

This is the value where we reward or punish the models. The properties of reward value is shown below:

* Positive value: Reward

* Negative Value: Punishment

* Large value: Large reward / punishment

* Small value: Small reward / punishment

## Action Labels

Action label is a label produced by the model. This label can be a part of decision-making classes or classification classes. For example:

* Decision-Making classes: "Up", "Down", "Left", "Right", "Forward", "Backward"

* Classification classes: 1, 2, 3, 4, 5, 6

## Reinforce Function

Upon calling reinforce() function, it will return two values, but we are interested in the first one for this tutorial.

```
local actionLabel = QLearningNeuralNetwork:reinforce(environmentFeatureVector, rewardValue)
```


