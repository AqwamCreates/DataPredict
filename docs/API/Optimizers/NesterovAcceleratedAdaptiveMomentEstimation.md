# [API Reference](../../API.md) - [Optimizers](../Optimizers.md) - NesterovAcceleratedAdaptiveMomentEstimation (NAdam)

## Constructors

### new()

Creates a new optimizer object. If there are no parameters given for that particular argument, then that argument will use default value.

```
NesterovAcceleratedAdaptiveMomentEstimation.new(Beta1: number, Beta2: number, Epsilon: number): OptimizerObject
```

#### Returns:

* OptimizerObject: The generated optimizer object.

## Functions

### setBeta1()

```
NesterovAcceleratedAdaptiveMomentEstimation:setBeta1(Beta1: number)
```

### setBeta2()

```
NesterovAcceleratedAdaptiveMomentEstimation:setBeta2(Beta2: number)
```

### setEpsilon()

```
AdaptiveMomentEstimationMaximum:setEpsilon(Epsilon: number)
```

### reset()

Reset optimizer's stored values (excluding the parameters).

```
AdaptiveMomentEstimation:reset()
```


