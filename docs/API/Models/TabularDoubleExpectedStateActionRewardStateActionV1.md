# [API Reference](../../API.md) - [Models](../Models.md) - TabularDoubleExpectedStateActionRewardStateActionV1 (Tabular Expected SARSA)

DeepExpectedStateActionRewardStateAction is a state-action grid with reinforcement learning capabilities. It can predict any positive numbers of discrete values.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
TabularDoubleExpectedStateActionRewardStateActionV1.new(learningRate: number, discountFactor: number, epsilon: number, EligibilityTrace: EligibilityTraceObject): ModelObject
```

#### Parameters:

* learningRate: The speed at which the model learns. Recommended that the value is set between 0 to 1. [Default: 0.1]

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1. [Default: 0.95]

* epsilon: Controls the balance between exploration and exploitation for calculating expected Q-values. The value must be set between 0 and 1. The value 0 focuses on exploitation only and 1 focuses on exploration only. [Default: 0.5]

* EligibilityTrace: The eligibility trace object to keep track of credit assignments of state-action pairs.

#### Returns:

* ModelObject: The generated model object.

## Inherited From

* [TabularReinforcementLearningBaseModel](TabularReinforcementLearningBaseModel.md)

## References

* [Expected SARSA in Reinforcement Learning](https://www.geeksforgeeks.org/expected-sarsa-in-reinforcement-learning/)

* [SARSA Reinforcement Learning](https://www.geeksforgeeks.org/sarsa-reinforcement-learning/)

* [Reinforcement Learning with Neural Network](https://www.baeldung.com/cs/reinforcement-learning-neural-network)
