# [API Reference](../../API.md) - [Optimizers](../Optimizers.md) - AdaptiveDelta (AdaDelta)

## Constructors

### new()

Creates a new optimizer object.

```
AdaptiveDelta.new(decayRate: number, weightDecayRate: number, epsilon: number): OptimizerObject
```

#### Parameters:

* decayRate: The value that controls the rate of decay.

* weightDecayRate: The value on how much we want the weights influence the gradient calculations. [Default: 0]

* epsilon: The value to ensure that the numbers are not divided by zero. [Default: 1e-16]

#### Returns:

* OptimizerObject: The generated optimizer object.

## Functions

### setDecayRate()

```
AdaptiveDelta:setDecayRate(decayRate: number)
```

#### Parameters:

* decayRate: The value that controls the rate of decay.

### setEpsilon()

```
AdaptiveDelta:setEpsilon(epsilon: number)
```

#### Parameters:

* epsilon: The value to ensure that the numbers are not divided by zero.

### reset()

Reset optimizer's stored values (excluding the parameters).

```
AdaptiveDelta:reset()
```

## Inherited From

* [BaseOptimizer](BaseOptimizer.md)
