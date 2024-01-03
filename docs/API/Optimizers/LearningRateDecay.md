# [API Reference](../../API.md) - [Optimizers](../Optimizers.md) - LearningRateDecay

## Constructors

### new()

Creates a new optimizer object.

```
LearningRateDecay.new(decayRate: number, timeStepToDecay: integer): OptimizerObject
```

#### Parameters:

* decayRate: The decay rate for learning rate.

* timeStepToDecay: The number of time steps to decay the learning rate.

#### Returns:

* OptimizerObject: The generated optimizer object.

## Functions

### setDecayRate()

```
LearningRateDecay:setDecayRate(decayRate: number)
```

#### Parameters:

* decayRate: The decay rate for learning rate.

### setTimeStepToDecay()

```
LearningRateDecay:setTimeStepToDecay(timeStepToDecay: integer)
```

#### Parameters:

* timeStepToDecay: The number of time steps to decay the learning rate.

### reset()

Reset optimizer's stored values (excluding the parameters).

```
AdaptiveGradient:reset()
```
