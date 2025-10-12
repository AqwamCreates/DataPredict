# [API Reference](../../API.md) - [Optimizers](../Optimizers.md) - RootMeanSquarePropagation (RMSProp)

## Constructors

### new()

Creates a new optimizer object. If there are no parameters given for that particular argument, then that argument will use default value.

```
RootMeanSquarePropagation.new(beta: number, epsilon: number, weightDecayRate: number): OptimizerObject
```

#### Parameters:

* beta: The value that controls the exponential decay rate for the moving average of squared gradients.

* weightDecayRate: The value on how much we want the weights influence the gradient calculations. [Default: 0]

* epsilon: The value to ensure that the numbers are not divided by zero. [Default: 1e-16]

#### Returns:

* OptimizerObject: The generated optimizer object.

## Functions

### setBeta()

```
RootMeanSquarePropagation:setBeta(beta: number)
```

#### Parameters:

* beta: The value that controls the exponential decay rate for the moving average of squared gradients.

### setEpsilon()

```
RootMeanSquarePropagation:setEpsilon(epsilon: number)
```

#### Parameters:

* epsilon: The value to ensure that the numbers are not divided by zero.

### reset()

Reset optimizer's stored values (excluding the parameters).

```
RootMeanSquarePropagation:reset()
```

## Inherited From

* [BaseOptimizer](BaseOptimizer.md)
