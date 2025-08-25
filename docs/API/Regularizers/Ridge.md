# [API Reference](../../API.md) - [Regularizers](../Regularizers.md) - Ridge (L2)

## Constructors

### new()

Creates a new regularizer object. If any of the arguments are not given, default argument values for that argument will be used.

```

Ridge.new({lambda: number}): RegularizerObject

```

#### Parameters:

* lambda: The regularization factor. Recommended values are between 0 to 1.

#### Returns:

* Regularizer: The generated regularizer object.