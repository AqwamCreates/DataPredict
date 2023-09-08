# [API Reference](../../API.md) - [Others](../Others.md) - Regularization

## Constructors

### new()

Creates a new regularization object. If any of the arguments are not given, default argument values for that argument will be used.

```
Regularization.new(lambda: number, regularisationMode: string, hasBias: boolean): RegularizationObject
```

#### Parameters:

* lambda: Regularization factor. Recommended values are between 0 to 1.

* regularisationMode: The mode which regularization will be used. Currently available ones are "L1" (or "Lasso"), "L2" (or "Ridge") and "L1+L2" (or "ElasticNet").

* hasBias: Set whether or not the regularization has bias.

#### Returns:

* RegularizationObject: The generated regularization object.

## Functions

### setParameters()

Set regularization’s parameters. When any of the arguments are not given, previous argument values for that argument will be used.

```
Regularization:setParameters(lambda: number, regularisationMode: string, hasBias: boolean)
```

#### Parameters:

* lambda: Regularization factor. Recommended values are between 0 to 1.

* regularisationMode: The mode which regularization will be used. Currently available ones are "L1" (or "Lasso"), "L2" (or "Ridge") and "L1+L2" (or "ElasticNet").

* hasBias: Set whether or not the regularization has bias.

### getLambda()

Get the lambda from the regularization object

```
Regularization:getLambda(): number
```

#### Returns:

* lambda: Regularization factor
