# [API Reference](../../API.md) - [Optimizers](../Optimizers.md) - AdaptiveGradientDelta (AdaDelta)

## Constructors

### new()

Creates a new optimizer object.

```
AdaptiveGradientDelta.new(DecayRate: number, Epsilon: number): OptimizerObject
```

#### Parameters:

* DecayRate: The rate of decay.

* Epsilon: The value to ensure that the numbers are not divided by zero.

#### Returns:

* OptimizerObject: The generated optimizer object.

## Functions

### setEpsilon()

```
AdaptiveGradientDelta:setEpsilon(Epsilon: number)
```

#### Parameters:

* Epsilon: The value to ensure that the numbers are not divided by zero.

### setDecayRate()

```
AdaptiveGradientDelta:setDecayRate(DecayRate: number)
```

#### Parameters:

* DecayRate: The rate of decay.

### reset()

Reset optimizer's stored values (excluding the parameters).

```
AdaptiveGradientDelta:reset()
```
