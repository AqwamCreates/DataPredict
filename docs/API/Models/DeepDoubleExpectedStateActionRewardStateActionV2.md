# [API Reference](../../API.md) - [Models](../Models.md) - DeepDoubleExpectedStateActionRewardStateActionV2 (Double Deep Expected SARSA)

DeepDoubleExpectedStateActionRewardStateActionV2 is a neural network with reinforcement learning capabilities. It can predict any positive numbers of discrete values. 

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
DeepDoubleExpectedStateActionRewardStateAction.new(epsilon: number, averagingRate: number, lambda: number, discountFactor: number): ModelObject
```

#### Parameters:

* epsilon: Controls the balance between exploration and exploitation for calculating expected Q-values. The value must be set between 0 and 1. The value 0 focuses on exploitation only and 1 focuses on exploration only. [Default 0.5]

* averagingRate: The higher the value, the faster the weights changes. The value must be set between 0 and 1. [Default: 0.995]

* lambda: At 0, the model acts like the Temporal Difference algorithm. At 1, the model acts as Monte Carlo algorithm. Between 0 and 1, the model acts as both. [Default: 0]

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1. [Default: 0.95]

#### Returns:

* ModelObject: The generated model object.

## Functions

## Inherited From

* [DeepReinforcementLearningBaseModel](DeepReinforcementLearningBaseModel.md)

## References

* [Double Sarsa and Double Expected Sarsa with Shallow and Deep Learning](https://www.scirp.org/journal/paperinformation.aspx?paperid=71237)

* [SARSA Reinforcement Learning](https://www.geeksforgeeks.org/sarsa-reinforcement-learning/)

* [Reinforcement Learning with Neural Network](https://www.baeldung.com/cs/reinforcement-learning-neural-network)
