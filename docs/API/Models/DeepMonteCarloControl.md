# [API Reference](../../API.md) - [Models](../Models.md) - DeepMonteCarloControl

DeepMonteCarloControl is a neural network with reinforcement learning capabilities. It can predict any positive numbers of discrete values.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
DeepMonteCarloControl.new(discountFactor: number): ModelObject
```

#### Parameters:

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1. [Default: 0.95]

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
DeepMonteCarloControl:setParameters(discountFactor: number)
```

#### Parameters:

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

## Inherited From

* [DeepReinforcementLearningBaseModel](DeepReinforcementLearningBaseModel.md)

## References

* [Forgetting Early Estimates in Monte Carlo Control Methods](https://ev.fe.uni-lj.si/3-2015/Vodopivec.pdf)
