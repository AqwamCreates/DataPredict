# [API Reference](../../API.md) - [Optimizers](../Optimizers.md) - LearningRateTimeDecay

## Constructors

### new()

Creates a new optimizer object.

```
LearningRateTimeDecay.new(decayRate: number, timeStepToDecay: integer): OptimizerObject
```

#### Parameters:

* decayRate: The decay rate for learning rate.

* timeStepToDecay: The number of time steps to decay the learning rate.

#### Returns:

* OptimizerObject: The generated optimizer object.

## Functions

### setDecayRate()

```
LearningRateTimeDecay:setDecayRate(decayRate: number)
```

#### Parameters:

* decayRate: The decay rate for learning rate.

### setTimeStepToDecay()

```
LearningRateTimeDecay:setTimeStepToDecay(timeStepToDecay: integer)
```

#### Parameters:

* timeStepToDecay: The number of time steps to decay the learning rate.

### reset()

Reset optimizer's stored values (excluding the parameters).

```
LearningRateTimeDecay:reset()
```

## Inherited From:

* [BaseOptimizer](BaseOptimizer.md)
