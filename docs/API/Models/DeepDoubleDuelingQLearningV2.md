# [API Reference](../../API.md) - [Models](../Models.md) - DeepDoubleDuelingQLearningV2 (D3QN)

DeepDoubleDuelingQLearningV2 is a neural network with reinforcement learning capabilities. It can predict any positive numbers of discrete values. 

It uses Hasselt et al. (2015) version, where it uses target and primary neural networks for training.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
DeepDoubleDuelingQLearning.new(averagingRate: number, discountFactor: number): ModelObject
```

#### Parameters:

* averagingRate: The higher the value, the faster the weights changes. The value must be set between 0 and 1.

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
DeepDoubleDuelingQLearning:setParameters(averagingRate: number, discountFactor: number)
```

#### Parameters:

* averagingRate: The higher the value, the faster the weights changes. The value must be set between 0 and 1.

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

## Inherited From

* [ReinforcementLearningDeepDuelingQLearningBaseModel](ReinforcementLearningDeepDuelingQLearningBaseModel.md)