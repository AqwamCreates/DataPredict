# [API Reference](../../API.md) - [Optimizers](../Optimizers.md) - AdaptiveMomentEstimation (Adam)

## Constructors

### new()

Creates a new optimizer object. If there are no parameters given for that particular argument, then that argument will use default value.

```
AdaptiveMomentEstimation.new(beta1: number, beta2: number, epsilon: number): OptimizerObject
```

#### Parameters:

* beta1: The decay rate of the moving average of the first moment of the gradients.

* beta2: The decay rate of the moving average of the squared gradients.

* epsilon: The value to ensure that the numbers are not divided by zero.

#### Returns:

* OptimizerObject: The generated optimizer object.

## Functions

### setBeta1()

```
AdaptiveMomentEstimation:setBeta1(beta1: number)
```
#### Parameters:

* beta1: The decay rate of the moving average of the first moment of the gradients.

### setBeta2()

```
AdaptiveMomentEstimation:setBeta2(beta2: number)
```
#### Parameters:

* beta2: The decay rate of the moving average of the squared gradients.

### setEpsilon()

```
AdaptiveMomentEstimation:setEpsilon(epsilon: number)
```

#### Parameters:

* epsilon: The value to ensure that the numbers are not divided by zero.

### reset()

Reset optimizer's stored values (excluding the parameters).

```
AdaptiveMomentEstimation:reset()
```
