# [API Reference](../../API.md) - [Regularizers](../Regularizers.md) - Lasso (L1)

## Constructors

### new()

Creates a new regularizer object. If any of the arguments are not given, default argument values for that argument will be used.

```

Lasso.new({lambda: number}): RegularizerObject

```

#### Parameters:

* lambda: The regularization factor. Recommended values are between 0 to 1.

#### Returns:

* Regularizer: The generated regularizer object.

## Inherited From

[BaseRegularizer](BaseRegularizer.md)
