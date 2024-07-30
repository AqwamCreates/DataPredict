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

### calculate()

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
BaseOptimizer:setCalculateFunction(calculateFunction: Function)
```

#### Parameters:

* The calculate function to be used by the base optimizer when calculate() function is called.

### setLearningRateScheduler()

Sets a scheduler for the learning rate.

```
BaseOptimizer:setLearningRateScheduler(LearningRateScheduler: ValueSchedulerObject)
```

#### Parameters:

# LearningRateScheduler: The value scheduler object to be used by the learning rate.

### setLearningRateScheduler()

Gets the scheduler for the learning rate.

```
BaseOptimizer:getLearningRateScheduler(): ValueSchedulerObject
```

#### Returns:

# LearningRateScheduler: The value scheduler object that was used by the learning rate.

### getOptimizerInternalParameters()

Gets the optimizer's internal parameters from the base optimizer.

```
BaseOptimizer:getOptimizerInternalParameters(doNotDeepCopy: boolean): {}
```

#### Parameters:

* doNotDeepCopy: Set whether or not to deep copy the optimizer internal parameters.

#### Returns:

* optimizerInternalParameters: A matrix/table containing optimizer internal parameters fetched from the base optimizer.

### setOptimizerInternalParameters()

Sets the optimizer's internal parameters from the base optimizer.

```
BaseOptimizer:setOptimizerInternalParameters(optimizerInternalParameters: {}, doNotDeepCopy: boolean)
```

#### Parameters:

* optimizerInternalParameters: A matrix/table containing optimizer internal parameters that will be used by the base optimizer.

* doNotDeepCopy: Set whether or not to deep copy the optimizer internal parameters.

### reset()

Reset optimizer's stored values (excluding the parameters).

```
BaseOptimizer:reset()
```
