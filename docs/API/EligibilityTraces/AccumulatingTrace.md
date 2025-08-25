# [API Reference](../../API.md) - [EligibilityTraces](../EligibilityTraces.md) - AccumulatingTrace

## Constructors

### new()

Creates a new eligibility trace object. If any of the arguments are not given, default argument values for that argument will be used.

```

Lasso.new({lambda: number}): EligibilityTraceObject

```

#### Parameters:

* lambda: The regularization factor. Recommended values are between 0 to 1.

#### Returns:

* EligibilityTraceObject: The generated eligibility trace object.
