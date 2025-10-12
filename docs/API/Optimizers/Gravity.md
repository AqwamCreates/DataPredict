# [API Reference](../../API.md) - [Optimizers](../Optimizers.md) - Gravity

## Constructors

### new()

Creates a new optimizer object. If there are no parameters given for that particular argument, then that argument will use default value.

```
Gravity.new(initialStepSize: number, movingAverage: number, weightDecayRate: number): OptimizerObject
```

#### Parameters:

* initialStepSize: The value to set the initial velocity during the first iteration.

* movingAverage: The value that controls the smoothing of gradients during training.

* weightDecayRate: The value on how much we want the weights influence the gradient calculations. [Default: 0]

#### Returns:

* OptimizerObject: The generated optimizer object.

## Functions

### setInitialStepSize()

```
Gravity:setDecayRate(initialStepSize: number)
```

#### Parameters:

* initialStepSize: The value to set the initial velocity during the first iteration.

### setMovingAverage()

```
Gravity:setDecayRate(movingAverage: number)
```

#### Parameters:

* movingAverage: The value that controls the smoothing of gradients during training.

### reset()

Reset optimizer's stored values (excluding the parameters).

```
Gravity:reset()
```

## Inherited From

* [BaseOptimizer](BaseOptimizer.md)
