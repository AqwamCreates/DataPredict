# [API Reference](../../API.md) - [Models](../Models.md) - (Deep Double Q Network)

DoubleQLearningNeuralNetworkV2 is a neural network with reinforcement learning capabilities. It can predict any positive numbers of discrete values. 

It uses Hasselt et al. (2015) version, where it uses target and primary neural networks for training.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
DeepDoubleQLearning.new(averagingRate: number, discountFactor: number, EligibilityTrace: EligibilityTraceObject): ModelObject
```

#### Parameters:

* averagingRate: The higher the value, the faster the weights changes. The value must be set between 0 and 1. [Default: 0.995]

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1. [Default: 0.95]

* EligibilityTrace: The eligibility trace object to keep track of credit assignments of state-action pairs.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
DeepDoubleQLearning:setParameters(averagingRate: number, lambda: number, discountFactor: number)
```

#### Parameters:

* averagingRate: The higher the value, the faster the weights changes. The value must be set between 0 and 1.

* lambda: At 0, the model acts like the Temporal Difference algorithm. At 1, the model acts as Monte Carlo algorithm.

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

## Inherited From

* [DeepReinforcementLearningBaseModel](DeepReinforcementLearningBaseModel.md)

## References

* [Double Deep Q Networks](https://towardsdatascience.com/double-deep-q-networks-905dd8325412)

* [Reinforcement Learning with Neural Network](https://www.baeldung.com/cs/reinforcement-learning-neural-network)

* [Reinforcement Q-Learning from Scratch in Python with OpenAI Gym](https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/)
