# [API Reference](../../API.md) - [Models](../Models.md) - TabularDoubleStateActionRewardStateActionV2 (Tabular Double SARSA)

TabularDoubleStateActionRewardStateActionV2 is a state-action grid with reinforcement learning capabilities. It can predict any positive numbers of discrete values.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
TabularDoubleStateActionRewardStateActionV2.new(discountFactor: number, averagingRate: number, EligibilityTrace: EligibilityTraceObject): ModelObject
```

#### Parameters:

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1. [Default: 0.95]

* averagingRate: The higher the value, the faster the weights changes. The value must be set between 0 and 1.

* EligibilityTrace: The eligibility trace object to keep track of credit assignments of state-action pairs.

#### Returns:

* ModelObject: The generated model object.

#### Parameters:

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

## Inherited From

* [TabularReinforcementLearningBaseModel](TabularReinforcementLearningBaseModel.md)

## References

* [SARSA Reinforcement Learning](https://www.geeksforgeeks.org/sarsa-reinforcement-learning/)

* [Reinforcement Learning with Neural Network](https://www.baeldung.com/cs/reinforcement-learning-neural-network)
