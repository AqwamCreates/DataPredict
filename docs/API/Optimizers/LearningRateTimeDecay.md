# [API Reference](../../API.md) - [Optimizers](../Optimizers.md) - LearningRateTimeDecay

## Constructors

### new()

Creates a new optimizer object. If there are no parameters given for that particular argument, then that argument will use default value.

```
LearningRateTimeDecay.new(decayRate: number): OptimizerObject
```

#### Parameters:

* decayRate: The decay rate for learning rate.

#### Returns:

* OptimizerObject: The generated optimizer object.

## Functions

### setDecayRate()

```
LearningRateTimeDecay:setDecayRate(decayRate: number)
```

#### Parameters:

* decayRate: The decay rate for learning rate.

## Inherited From

* [BaseOptimizer](BaseOptimizer.md)
