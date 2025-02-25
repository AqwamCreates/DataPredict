# [API Reference](../../API.md) - [Models](../Models.md) - BaseModel

The base model for all machine and deep learning models.

## Constructors

### new()

Creates a new machine learning base model.

```
IterativeMethodBaseModel.new(): BaseModelObject
```

## Functions

### setNumberOfIterationsPerCostCalculation()

Set the number of iterations needed to calculate the costs. This is to save computational time.

```
BaseModel:setModelParameters(numberOfIterationsPerCostCalculation: number)
```

#### Parameters

* numberOfIterationsPerCostCalculation: The number of iterations for each cost calculation.

### setNumberOfIterationsToCheckIfConverged()

Set the number of iterations needed to confirm convergence.

```
IterativeMethodBaseModel:setNumberOfIterationsToCheckIfConverged(numberOfIterations: number)
```

#### Parameters

* numberOfIterations: The number of iterations for confirming convergence.

### setTargetCost()

Set the upper bound and lower bounds of the target cost.
```
IterativeMethodBaseModel:setTargetCost(upperBound: number, lowerBound: number)
```

#### Parameters

* upperBound: The upper bound of target cost.

* lowerBound: The lower bound of target cost.

### setWaitDurations()

Set wait durations inside the models to avoid exhausting script running time.

```
IterativeMethodBaseModel:setWaitDurations(iterationWaitDuration: number, dataWaitDuration: number, sequenceWaitDuration: number)
```

#### Parameters:

* iterationWaitDuration: The wait duration between the iterations.

* dataWaitDuration: The wait duration between the data calculations.

* sequenceWaitDuration: The wait duration between the token sequence.

### Inherited From

* [BaseModel](BaseModel.md)
