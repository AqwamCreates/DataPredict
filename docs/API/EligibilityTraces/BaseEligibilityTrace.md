# [API Reference](../../API.md) - [EligibilityTraces](../EligibilityTraces.md) - BaseEligibilityTrace

## Constructors

### new()

Creates a new eligibility trace object. If any of the arguments are not given, default argument values for that argument will be used.

```

BaseEligibilityTrace.new({lambda: number}): EligibilityTraceObject

```

#### Parameters:

* lambda: lambda: At 0, the model acts like the Temporal Difference algorithm. At 1, the model acts as Monte Carlo algorithm. Between 0 and 1, the model acts as both. [Default: 0]

#### Returns:

* EligibilityTraceObject: The generated eligibility trace object.

## Functions

### increment()

```

BaseEligibilityTrace:increment(actionIndex: number, discountFactor: number, dimensionSizeArray: {number})

```

#### Parameters:

* actionIndex: The action index to be incremented.

* discountFactor: The discount factor to be used to modify the eligibility trace.

* dimensionSizeArray: The dimension size array for generating the eligibility trace.

### calculate()

```

BaseEligibilityTrace:calculate(temporalDifferenceErrorVector: tensor): tensor

```

#### Parameters:

* temporalDifferenceErrorVector: A temporal difference error vector.

#### Returns:

* temporalDifferenceErrorVector: A temporal difference error vector.

### setCalculateFunction()

```

BaseEligibilityTrace:setIncrementFunction(IncrementFunction: function)

```

#### Parameters:

* IncrementFunction: The increment function to be set.

### setLambda()

Set the lambda to the eligibility trace object.

```

BaseEligibilityTrace:setLambda(lambda: number)

```

#### Parameters:

* lambda: The regularization factor. Recommended values are between 0 to 1.

### getLambda()

Get the lambda from the eligibility trace object.

```

BaseEligibilityTrace:getLambda(): number

```

#### Returns:

* lambda: The regularization factor.

### reset()

Resets the stored eligibility traces.

```

BaseEligibilityTrace:reset()

```

## Inherited From

[BaseInstance](../Cores/BaseInstance.md)
