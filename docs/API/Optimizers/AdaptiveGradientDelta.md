# [API Reference](../../API.md) - [Optimizers](../Optimizers.md) - AdaptiveGradientDelta (AdaDelta)

## Constructors

### new()

Creates a new optimizer object.

```
AdaptiveGradientDelta.new(decayRate: number, epsilon: number): OptimizerObject
```

#### Parameters:

* decayRate: The value that controls the rate of decay.

* epsilon: The value to ensure that the numbers are not divided by zero.

#### Returns:

* OptimizerObject: The generated optimizer object.

## Functions

### setDecayRate()

```
AdaptiveGradientDelta:setDecayRate(decayRate: number)
```

#### Parameters:

* decayRate: The value that controls the rate of decay.

### setEpsilon()

```
AdaptiveGradientDelta:setEpsilon(epsilon: number)
```

#### Parameters:

* epsilon: The value to ensure that the numbers are not divided by zero.

### reset()

Reset optimizer's stored values (excluding the parameters).

```
AdaptiveGradientDelta:reset()
```

## Inherited From:

* [BaseOptimizer](BaseOptimizer.md)
