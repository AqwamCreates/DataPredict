# [API Reference](../../API.md) - [Optimizers](../Optimizers.md) - RootMeanSquarePropagation (RMSProp)

## Constructors

### new()

Creates a new optimizer object. If there are no parameters given for that particular argument, then that argument will use default value.

```
RootMeanSquarePropagation.new(Beta: Number, Epsilon: Number): OptimizerObject
```

#### Returns:

* OptimizerObject: The generated optimizer object.

## Functions

### setBeta()

```
RootMeanSquarePropagation:setBeta(Beta: Number)
```

### setEpsilon()

```
RootMeanSquarePropagation:setEpsilon(Epsilon: Number)
```

### reset()

Reset optimizer's stored values (excluding the parameters).

```
RootMeanSquarePropagation:reset()
```

## Notes:

* Epsilon is used to prevent a numerator from dividing by zero. Otherwise, the resulting calculation would be infinity.

* Generally, the epsilon values are usually set to very small positive decimal numbers.

* If you are unsure how epsilon works, then leave the setting to default.
