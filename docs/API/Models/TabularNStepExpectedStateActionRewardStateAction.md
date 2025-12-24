# [API Reference](../../API.md) - [Models](../Models.md) - TabularNStepExpectedStateActionRewardStateAction (Tabular N-Step Expected SARSA)

TabularNStepExpectedStateActionRewardStateAction is a state-action grid with reinforcement learning capabilities. It can predict any positive numbers of discrete values.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
TabularNStepExpectedStateActionRewardStateAction.new(epsilon: number, nStep: number, discountFactor: number): ModelObject
```

#### Parameters:

* epsilon: Controls the balance between exploration and exploitation for calculating expected Q-values. The value must be set between 0 and 1. The value 0 focuses on exploitation only and 1 focuses on exploration only. [Default: 0.5]

* nStep: The number of future steps considered for credit assignment. Higher values extend the planning horizon but increase variance. Set to 1 for standard expected SARSA. [Default: 3]

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1. [Default: 0.95]

#### Returns:

* ModelObject: The generated model object.

## Inherited From

* [TabularReinforcementLearningBaseModel](TabularReinforcementLearningBaseModel.md)

## References

* [Expected SARSA in Reinforcement Learning](https://www.geeksforgeeks.org/expected-sarsa-in-reinforcement-learning/)

* [SARSA Reinforcement Learning](https://www.geeksforgeeks.org/sarsa-reinforcement-learning/)

* [Reinforcement Learning with Neural Network](https://www.baeldung.com/cs/reinforcement-learning-neural-network)
