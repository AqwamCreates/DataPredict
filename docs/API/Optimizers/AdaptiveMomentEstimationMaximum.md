# [API Reference](../../API.md) - [Optimizers](../Optimizers.md) - AdaptiveMomentEstimationMaximum (AdaMax)

## Constructors

### new()

Creates a new optimizer object. If there are no parameters given for that particular argument, then that argument will use default value.

```
AdaptiveMomentEstimationMaximum.new(Beta1: number, Beta2: number, Epsilon: number): OptimizerObject
```
#### Parameters:

* Beta1: The decay rate of the moving average of the first moment of the gradients.

* Beta2: The decay rate of the moving average of the squared gradients.

* Epsilon: The value to ensure that the numbers are not divided by zero.

#### Returns:

* OptimizerObject: The generated optimizer object.

## Functions

### setBeta1()

```
AdaptiveMomentEstimationMaximum:setBeta1(Beta1: number)
```

#### Parameters:

* Beta1: The decay rate of the moving average of the first moment of the gradients.

### setBeta2()

```
AdaptiveMomentEstimationMaximum:setBeta2(Beta2: number)
```

#### Parameters:

* Beta2: The decay rate of the moving average of the squared gradients.

### setEpsilon()

```
AdaptiveMomentEstimationMaximum:setEpsilon(Epsilon: number)
```

#### Parameters:

* Epsilon: The value to ensure that the numbers are not divided by zero.

### reset()

Reset optimizer's stored values (excluding the parameters).

```
AdaptiveMomentEstimationMaximum:reset()
```

## Inherited From:

* [BaseOptimizer](BaseOptimizer.md)
