# [API Reference](../../API.md) - [Optimizers](../Optimizers.md) - AdaptiveMomentEstimation (Adam)

## Constructors

### new()

Creates a new optimizer object. If there are no parameters given for that particular argument, then that argument will use default value.

```
AdaptiveMomentEstimation.new(Beta1: number, Beta2: number, Epsilon: number): OptimizerObject
```

#### Returns:

* OptimizerObject: The generated optimizer object.

## Functions

### setBeta1()

```
AdaptiveMomentEstimation:setBeta1(Beta1: number)
```

### setBeta2()

```
AdaptiveMomentEstimation:setBeta2(Beta2: number)
```

### setEpsilon()

```
AdaptiveMomentEstimation:setEpsilon(Epsilon: number)
```

#### Parameters:

* Epsilon: The value to ensure that the numbers are not divided by zero.

### reset()

Reset optimizer's stored values (excluding the parameters).

```
AdaptiveMomentEstimation:reset()
```
