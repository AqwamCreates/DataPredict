# [API Reference](../../API.md) - [Optimizers](../Optimizers.md) - AdaptiveGradientDelta (AdaDelta)

## Constructors

### new()

Creates a new optimizer object.

```
AdaptiveGradientDelta.new(DecayRate: number, Epsilon: number): OptimizerObject
```

#### Parameters:

* DecayRate: The value that controls the rate of decay.

* Epsilon: The value to ensure that the numbers are not divided by zero.

#### Returns:

* OptimizerObject: The generated optimizer object.

## Functions

### setDecayRate()

```
AdaptiveGradientDelta:setDecayRate(DecayRate: number)
```

#### Parameters:

* DecayRate: The value that controls the rate of decay.

### setEpsilon()

```
AdaptiveGradientDelta:setEpsilon(Epsilon: number)
```

#### Parameters:

* Epsilon: The value to ensure that the numbers are not divided by zero.

### reset()

Reset optimizer's stored values (excluding the parameters).

```
AdaptiveGradientDelta:reset()
```
