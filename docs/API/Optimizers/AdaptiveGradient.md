# [API Reference](../../API.md) - [Optimizers](../Optimizers.md) - AdaptiveGradient (Adagrad)

## Constructors

### new()

Creates a new optimizer object.

```
AdaptiveGradient.new(weightDecayRate: number): OptimizerObject
```

### Parameters:

* weightDecayRate: The value on how much we want the weights influence the gradient calculations. [Default: 0]

#### Returns:

* OptimizerObject: The generated optimizer object.

## Functions

### reset()

Reset optimizer's stored values (excluding the parameters).

```
AdaptiveGradient:reset()
```

## Inherited From

* [BaseOptimizer](BaseOptimizer.md)
