# [API Reference](../../API.md) - [Optimizers](../Optimizers.md) - Momentum

## Constructors

### new()

Creates a new optimizer object. If there are no parameters given for that particular argument, then that argument will use default value.

```
Momentum.new(decayRate: number, weightDecayRate: number): OptimizerObject
```
#### Parameters:

* decayRate: The value that controls the rate of decay.

* weightDecayRate: The value on how much we want the weights influence the gradient calculations. [Default: 0]

#### Returns:

* OptimizerObject: The generated optimizer object.

## Functions

### setDecayRate()

```
Momentum:setDecayRate(decayRate: number)
```

#### Parameters:

* decayRate: The value that controls the rate of decay.

### reset()

Reset optimizer's stored values (excluding the parameters).

```
Momentum:reset()
```

## Inherited From

* [BaseOptimizer](BaseOptimizer.md)
