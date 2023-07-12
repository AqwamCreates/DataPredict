# [API Reference](../../API.md) - [Optimizers](../Optimizers.md) - AdaptiveGradientDelta (AdaDelta)

## Constructors

### new()

Creates a new optimizer object.

```
AdaptiveGradient.new(DecayRate: number, Epsilon: number): OptimizerObject
```

#### Returns:

* OptimizerObject: The generated optimizer object.

## Functions

### reset()

Reset optimizer's stored values (excluding the parameters).

```
AdaptiveGradient:reset()
```
