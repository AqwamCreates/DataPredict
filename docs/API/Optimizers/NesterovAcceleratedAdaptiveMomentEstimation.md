# [API Reference](../../API.md) - [Optimizers](../Optimizers.md) - NesterovAcceleratedAdaptiveMomentEstimation (NAdam)

## Constructors

### new()

Creates a new optimizer object. If there are no parameters given for that particular argument, then that argument will use default value.

```
NesterovAcceleratedAdaptiveMomentEstimation.new(Beta1: number, Beta2: number, Epsilon: number): OptimizerObject
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
NesterovAcceleratedAdaptiveMomentEstimation:setBeta1(Beta1: number)
```

#### Parameters:

* Beta1: The decay rate of the moving average of the first moment of the gradients.

### setBeta2()

```
NesterovAcceleratedAdaptiveMomentEstimation:setBeta2(Beta2: number)
```

#### Parameters:

* Beta2: The decay rate of the moving average of the squared gradients.

### setEpsilon()

```
NesterovAcceleratedAdaptiveMomentEstimation:setEpsilon(Epsilon: number)
```

#### Parameters:

* Epsilon: The value to ensure that the numbers are not divided by zero.

### reset()

Reset optimizer's stored values (excluding the parameters).

```
NesterovAcceleratedAdaptiveMomentEstimation:reset()
```

## Notes:

* Epsilon is used to prevent a numerator from dividing by zero. Otherwise, the resulting calculation would be infinity.

* Generally, the epsilon values are usually set to very small positive decimal numbers.

* If you are unsure how epsilon works, then leave the setting to default.
