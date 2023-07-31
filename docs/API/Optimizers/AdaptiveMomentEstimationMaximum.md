# [API Reference](../../API.md) - [Optimizers](../Optimizers.md) - AdaptiveMomentEstimationMaximum (AdaMax)

## Constructors

### new()

Creates a new optimizer object. If there are no parameters given for that particular argument, then that argument will use default value.

```
AdaptiveMomentEstimationMaximum.new(Beta1: number, Beta2: number, Epsilon: number): OptimizerObject
```

#### Returns:

* OptimizerObject: The generated optimizer object.

## Functions

### setBeta1()

```
AdaptiveMomentEstimationMaximum:setBeta1(Beta1: number)
```

### setBeta2()

```
AdaptiveMomentEstimationMaximum:setBeta2(Beta2: number)
```

### setEpsilon()

```
AdaptiveMomentEstimationMaximum:setEpsilon(Epsilon: number)
```

### reset()

Reset optimizer's stored values (excluding the parameters).

```
AdaptiveMomentEstimationMaximum:reset()
```

## Notes:

* Epsilon is used to prevent a numerator from dividing by zero. Otherwise, the resulting calculation would be infinity.

* Generally, the epsilon values are usually set to very small positive decimal numbers.

* If you are unsure how epsilon works, then leave the setting to default.
