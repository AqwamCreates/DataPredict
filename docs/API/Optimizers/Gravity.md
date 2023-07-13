# [API Reference](../../API.md) - [Optimizers](../Optimizers.md) - Gravity

## Constructors

### new()

Creates a new optimizer object. If there are no parameters given for that particular argument, then that argument will use default value.

```
Gravity.new(InitialStepSize: number, MovingAverage: number): OptimizerObject
```

#### Returns:

* OptimizerObject: The generated optimizer object.

## Functions

### setInitialStepSize()

```
Gravity:setDecayRate(InitialStepSize: number)
```

### setMovingAverage()

```
Gravity:setDecayRate(MovingAverage: number)
```

### reset()

Reset optimizer's stored values (excluding the parameters).

```
Gravity:reset()
```
