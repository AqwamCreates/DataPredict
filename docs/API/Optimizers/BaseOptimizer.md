# [API Reference](../../API.md) - [Optimizers](../Optimizers.md) - BaseOptimizer

BaseOptimizer is a base for all optimizers.

## Constructors

### new()

Creates a new base optimizer object.

```
BaseOptimizer.new(optimizerName: string): BaseOptimizerObject
```

#### Parameters

* optimizerName: The optimizer name to inherit the base class.

#### Returns:

* OptimizerObject: The generated optimizer object.

## Functions

## calculate()

Returns a modified cost function derivatives.

```
BaseOptimizer:calculate(learningRate: number, costFunctionDerivatives: matrix): matrix
```

#### Parameters:

* learningRate: The learning rate used by a model.

* costFunctionDerivatives: The cost function derivatives calculated by a model.

#### Returns:

* costFunctionDerivatives: The modified cost function derivatives that is to be used by a model.

### reset()

Reset optimizer's stored values (excluding the parameters).

```
BaseOptimizer:reset()
```

### setCalculateFunction()

Sets a calculate function for the base optimizer.

```
BaseOptimizer:setCalculateFunction(calculateFunction)
```

#### Parameters:

* The calculate function to be used by the base optimizer when calculate() function is called.

### setResetFunction()

Sets a reset function for the base optimizer.

```
BaseOptimizer:setResetFunction(calculateFunction)
```

#### Parameters:

* The reset function to be used by the base optimizer when reset() function is called.
