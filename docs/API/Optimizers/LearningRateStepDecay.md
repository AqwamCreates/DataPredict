# [API Reference](../../API.md) - [Optimizers](../Optimizers.md) - LearningRateStepDecay

## Constructors

### new()

Creates a new optimizer object. If there are no parameters given for that particular argument, then that argument will use default value.

```
LearningRateStepDecay.new(decayRate: number, timeStepToDecay: integer): OptimizerObject
```

#### Parameters:

* timeStepToDecay: The number of time steps to decay the learning rate.

* decayRate: The decay rate for learning rate.

#### Returns:

* OptimizerObject: The generated optimizer object.

## Functions

### setDecayRate()

```
LearningRateStepDecay:setDecayRate(decayRate: number)
```

#### Parameters:

* decayRate: The decay rate for learning rate.

### setTimeStepToDecay()

```
LearningRateStepDecay:setTimeStepToDecay(timeStepToDecay: integer)
```

#### Parameters:

* timeStepToDecay: The number of time steps to decay the learning rate.

### reset()

Reset optimizer's stored values (excluding the parameters).

```
LearningRateStepDecay:reset()
```

## Inherited From

* [BaseOptimizer](BaseOptimizer.md)
