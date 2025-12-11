# [API Reference](../../API.md) - [Models](../Models.md) - TabularNStepStateActionRewardStateAction (Tabular N-Step SARSA)

TabularNStepStateActionRewardStateAction is a state-action grid with reinforcement learning capabilities. It can predict any positive numbers of discrete values.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
TabularNStepStateActionRewardStateAction.new(learningRate: number, nStep: number, discountFactor: number): ModelObject
```

#### Parameters:

* learningRate: The speed at which the model learns. Recommended that the value is set between 0 to 1. [Default: 0.1]

* nStep: The number of future steps considered for credit assignment. Higher values extend the planning horizon but increase variance. Set to 1 for standard Q-learning. [Default: 3]

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1. [Default: 0.95]

#### Returns:

* ModelObject: The generated model object.

## Inherited From

* [TabularReinforcementLearningBaseModel](TabularReinforcementLearningBaseModel.md)

## References

* [SARSA Reinforcement Learning](https://www.geeksforgeeks.org/sarsa-reinforcement-learning/)

* [Reinforcement Learning with Neural Network](https://www.baeldung.com/cs/reinforcement-learning-neural-network)
