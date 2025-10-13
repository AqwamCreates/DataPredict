# [API Reference](../../API.md) - [Models](../Models.md) - TabularMonteCarloControl

TabularMonteCarloControl is a state-action grid with reinforcement learning capabilities. It can predict any positive numbers of discrete values.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
TabularMonteCarloControl.new(learningRate: number, discountFactor: number): ModelObject
```

#### Parameters:

* learningRate: The speed at which the model learns. Recommended that the value is set between 0 to 1. [Default: 0.1]

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1. [Default: 0.95]

#### Returns:

* ModelObject: The generated model object.

## Inherited From

* [TabularReinforcementLearningBaseModel](TabularReinforcementLearningBaseModel.md)

## References

* [Forgetting Early Estimates in Monte Carlo Control Methods](https://ev.fe.uni-lj.si/3-2015/Vodopivec.pdf)
