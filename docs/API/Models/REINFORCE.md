# [API Reference](../../API.md) - [Models](../Models.md) - REINFORCE

REINFORCE is a neural network with reinforcement learning capabilities. It can predict any positive numbers of discrete values.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
REINFORCE.new(discountFactor: number): ModelObject
```

#### Parameters:

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1. [Default: 0.95]

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
REINFORCE:setParameters(discountFactor: number)
```

#### Parameters:

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

## Inherited From

* [ReinforcementLearningBaseModel](ReinforcementLearningBaseModel.md)

## References

* [REINFORCE – A Quick Introduction (with Code)](https://dilithjay.com/blog/reinforce-a-quick-introduction-with-code/)

* [REINFORCE — a policy-gradient based reinforcement Learning algorithm](https://medium.com/intro-to-artificial-intelligence/reinforce-a-policy-gradient-based-reinforcement-learning-algorithm-84bde440c816)

* [REINFORCE](https://paperswithcode.com/method/reinforce)
