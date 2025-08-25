# [API Reference](../../API.md) - [EligibilityTraces](../EligibilityTraces.md) - AccumulatingTrace

## Constructors

### new()

Creates a new eligibility trace object. If any of the arguments are not given, default argument values for that argument will be used.

```

AccumulatingTrace.new({lambda: number}): EligibilityTraceObject

```

#### Parameters:

* lambda: lambda: At 0, the model acts like the Temporal Difference algorithm. At 1, the model acts as Monte Carlo algorithm. Between 0 and 1, the model acts as both. [Default: 0]

#### Returns:

* EligibilityTraceObject: The generated eligibility trace object.

## Inherited From

[BaseEligibilityTrace](BaseEligibilityTrace.md)
