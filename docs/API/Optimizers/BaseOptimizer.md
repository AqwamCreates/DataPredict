# [API Reference](../../API.md) - [Optimizers](../Optimizers.md) - BaseOptimizer

BaseOptimizer is a base for all optimizers.

## Constructors

### new()

Creates a new base optimizer object.

```
BaseOptimizer.new(optimizerName: string): BaseOptimizerObject
```

#### Parameters

* optimizerName: The optimizer name that is stored in base optimizer.

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

### getOptimizerName()

Gets the optimizer's name from the base optimizer.

```
BaseOptimizer:getOptimizerName()
```

#### Returns:

* optimizerName: The optimizer name that is stored in base optimizer.

### setCalculateFunction()

Sets a calculate function for the base optimizer.

```
BaseOptimizer:setCalculateFunction(calculateFunction)
```

#### Parameters:

* The calculate function to be used by the base optimizer when calculate() function is called.

### getOptimizerInternalParameters()

Gets the optimizer's internal parameters from the base optimizer.

```
BaseOptimizer:getOptimizerInternalParameters(doNotDeepCopy: boolean): {}
```

#### Parameters:

* doNotDeepCopy: Set whether or not to deep copy the optimizer internal parameters.

#### Returns:

* optimizerInternalParameters: The optimizer internal parameters that is stored in base optimizer.

### setOptimizerInternalParameters()

Sets the optimizer's internal parameters from the base optimizer.

```
BaseOptimizer:setOptimizerInternalParameters(optimizerInternalParameters: {}, doNotDeepCopy: boolean)
```

#### Parameters:

* optimizerInternalParameters: The optimizer internal parameters that is stored to be stored in base optimizer.

* doNotDeepCopy: Set whether or not to deep copy the optimizer internal parameters.

### reset()

Reset optimizer's stored values (excluding the parameters).

```
BaseOptimizer:reset()
```
