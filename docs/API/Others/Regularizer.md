# [API Reference](../../API.md) - [Others](../Others.md) - Regularizer

## Constructors

### new()

Creates a new regularization object. If any of the arguments are not given, default argument values for that argument will be used.

```
Regularizer.new(lambda: number, regularizationMode: string, hasBias: boolean): RegularizationObject
```

#### Parameters:

* lambda: The regularization factor. Recommended values are between 0 to 1.

* regularizationMode: The mode which regularization will be used. Currently available ones are "L1" (or "Lasso"), "L2" (or "Ridge") and "L1+L2" (or "ElasticNet").

* hasBias: Set whether or not the regularization has bias.

#### Returns:

* RegularizationObject: The generated regularization object.

## Functions

### setParameters()

Set regularization’s parameters. When any of the arguments are not given, previous argument values for that argument will be used.

```
Regularizer:setParameters(lambda: number, regularizationMode: string, hasBias: boolean)
```

#### Parameters:

* lambda: The regularization factor. Recommended values are between 0 to 1.

* regularisationMode: The mode which regularization will be used. Currently available ones are "L1" (or "Lasso"), "L2" (or "Ridge") and "L1+L2" (or "ElasticNet").

* hasBias: Set whether or not the regularization has bias.

### getLambda()

Get the lambda from the regularization object.

```
Regularizer:getLambda(): number
```

#### Returns:

* lambda: The regularization factor.

## Inherited From

* [BaseInstance](../Cores/BaseInstance.md)