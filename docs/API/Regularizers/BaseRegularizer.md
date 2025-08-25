# [API Reference](../../API.md) - [Regularizers](../Regularizers.md) - BaseRegularizer

## Constructors

### new()

Creates a new regularizer object. If any of the arguments are not given, default argument values for that argument will be used.

```

BaseRegularizer.new({lambda: number}): RegularizerObject

```

#### Parameters:

* lambda: The regularization factor. Recommended values are between 0 to 1.

#### Returns:

* Regularizer: The generated regularizer object.

## Functions

### calculate()

```

BaseRegularizer:calculate(weightTensor: tensor)

```

#### Parameters:

* weightTensor: A tensor from weight block object.

#### Returns:

* weightTensor: A tensor from weight block object.

### setCalculateFunction()

```

BaseRegularizer:setCalculateFunction(calculateFunction: function)

```

#### Parameters:

* calculateFunction: The calculate function to be set.

### setLambda()

Set the lambda to the regularizer object.

```

BaseRegularizer:setLambda(lambda: number)

```

#### Parameters:

* lambda: The regularization factor. Recommended values are between 0 to 1.

### getLambda()

Get the lambda from the regularizer object.

```

BaseRegularizer:getLambda(): number

```

#### Returns:

* lambda: The regularization factor.

## Inherited From

[BaseInstance](../Cores/BaseInstance.md)
