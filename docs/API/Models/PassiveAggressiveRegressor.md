# [API Reference](../../API.md) - [Models](../Models.md) - PassiveAggressiveRegressor

PassiveAggressiveRegressor is a supervised machine learning model that predicts continuous values (e.g. 1.2, -32, 90, -1.2 and etc. ). It uses iterative calculations to find the best model parameters.

## Stored Model Parameters

Contains a matrix.  

* ModelParameters[I][J]: Value of matrix at row I and column J. The rows are the features.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
PassiveAggressiveRegressor.new(maximumNumberOfIterations: integer, variant: string, epsilon: number, cValue: number): ModelObject
```

#### Parameters:

* maximumNumberOfIterations: How many times should the model needed to be trained. [Default: 500]

* variant: Controls which PassiveAggressiveRegressor variant to use. Available options are:

    * 0 (Default)
 
    * 1
 
    * 2

* epsilon: Controls the epsilon-insensitive margin. Higher values make the model less sensitive to small deviations. [Default: 0]

* cValue: The aggressiveness parameter used in some Passive-Aggressive variants (1 and 2). Higher values allow larger updates per misclassified example, while lower values limit the update magnitude. [Default: 1]

#### Returns:

* ModelObject: The generated model object.

## Functions

### train()

Train the model.

```
PassiveAggressiveRegressor:train(featureMatrix: Matrix, labelVector: Matrix): number[]
```

#### Parameters:

* featureMatrix: Matrix containing all data.

* labelVector: A (n x 1) matrix containing values related to featureMatrix.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict the values for given data.

```
PassiveAggressiveRegressor:predict(featureMatrix: Matrix): Matrix
```

#### Parameters:

* featureMatrix: Matrix containing data.

#### Returns:

* predictedValueVector: A vector containing the values that is predicted by the model.

## Inherited From

* [IterativeMethodBaseModel](IterativeMethodBaseModel.md)

## References

[Online Passive-Aggressive Algorithms](https://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf)
