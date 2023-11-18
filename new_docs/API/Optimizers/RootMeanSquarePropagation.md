# [API Reference](../../API.md) - [Optimizers](../Optimizers.md) - RootMeanSquarePropagation (RMSProp)

## Constructors

### new()

Creates a new optimizer object. If there are no parameters given for that particular argument, then that argument will use default value.

```
RootMeanSquarePropagation.new(Beta: number, Epsilon: number): OptimizerObject
```

#### Parameters:

* Beta: The value that controls the exponential decay rate for the moving average of squared gradients.

* Epsilon: The value to ensure that the numbers are not divided by zero.

#### Returns:

* OptimizerObject: The generated optimizer object.

## Functions

### setBeta()

```
RootMeanSquarePropagation:setBeta(Beta: number)
```

#### Parameters:

* Beta: The value that controls the exponential decay rate for the moving average of squared gradients.

### setEpsilon()

```
RootMeanSquarePropagation:setEpsilon(Epsilon: number)
```

#### Parameters:

* Epsilon: The value to ensure that the numbers are not divided by zero.

### reset()

Reset optimizer's stored values (excluding the parameters).

```
RootMeanSquarePropagation:reset()
```
