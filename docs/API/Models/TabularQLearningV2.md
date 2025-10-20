# [API Reference](../../API.md) - [Models](../Models.md) - TabularDoubleQLearningV2 (Tabular Double Q-Learning)

TabularDoubleQLearningV2 is a state-action grid with reinforcement learning capabilities. It can predict any positive numbers of discrete values.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
TabularDoubleQLearningV2.new(learningRate: number, discountFactor: number, averagingRate: number, EligibilityTrace: EligibilityTraceObject): ModelObject
```

#### Parameters:

* learningRate: The speed at which the model learns. Recommended that the value is set between 0 to 1. [Default: 0.1]

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
