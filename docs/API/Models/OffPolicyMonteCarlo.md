# [API Reference](../../API.md) - [Models](../Models.md) - OffPolicyMonteCarlo

OffPolicyMonteCarlo is a neural network with reinforcement learning capabilities. It can predict any positive numbers of discrete values.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
OffPolicyMonteCarlo.new(behaviourPolicyFunction: string, discountFactor: number): ModelObject
```

#### Parameters:

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
OffPolicyMonteCarlo:setParameters(behaviourPolicyFunction: string, discountFactor: number)
```

#### Parameters:

* behaviourPolicyFunction: 

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

## Inherited From

* [ReinforcementLearningBaseModel](ReinforcementLearningBaseModel.md)

## References

* [Off-Policy Monte Carlo Control](http://incompleteideas.net/book/first/ebook/node56.html)

* [Forgetting Early Estimates in Monte Carlo Control Methods](https://ev.fe.uni-lj.si/3-2015/Vodopivec.pdf)
