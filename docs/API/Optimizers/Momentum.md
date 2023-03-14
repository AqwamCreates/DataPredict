# [API Reference](../../API.md) - [Optimizers](../Optimizers.md) - Momentum

## Constructors

### new()

Creates a new optimizer object. If there are no parameters given for that particular argument, then that argument will use default value.

```
Momentum.new(DecayRate: Number): OptimizerObject
```

#### Returns:

* OptimizerObject: The generated optimizer object.

## Functions

### setDecayRate()

```
Momentum:setDecayRate(DecayRate: Number)
```

### reset()

Reset optimizer's stored values (excluding the parameters).

```
Momentum:reset()
```
