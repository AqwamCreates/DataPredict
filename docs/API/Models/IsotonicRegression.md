# [API Reference](../../API.md) - [Models](../Models.md) - IsotonicRegression

IsotonicRegression is a supervised machine learning model that predicts average values (e.g. 1.2, -32, 90, -1.2 and etc.) if it falls within certain ranges. It uses grouped average calculations to find the best model parameters.

This model can only accept feature matrices and label vectors that has 1 column.

## Stored Model Parameters

Contains a matrix.  

* ModelParameters[I][J]: Value of matrix at row I and column J. The rows are the features.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
IsotonicRegression.new(isIncreasing: boolean, onOutOfBounds: string): ModelObject
```

#### Parameters:

* isIncreasing: Set whether or not if the label values should increase as feature values increases. Otherwise, the label values would decrease as the feature value increases. [Default: true]

* onOutOfBounds: Set on how to handle feature values that is out of model parameters' bounds when calling predict() function. Available options includes:

  * NotANumber (Default)

  * Clamp

#### Returns:

* ModelObject: The generated model object.

## Functions

### train()

Train the model.

```
IsotonicRegression:train(featureMatrix: Matrix, labelVector: Matrix): number[]
```

#### Parameters:

* featureMatrix: Matrix containing all data. It must only have one column.

* labelVector: A (n x 1) matrix containing values related to featureMatrix. It must only have one column.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict the value for a given data.

```
IsotonicRegression:predict(featureMatrix: Matrix): Matrix
```

#### Parameters:

* featureMatrix: Matrix containing data. It must only have one column.

#### Returns:

* predictedVector: A vector containing values that are predicted by the model.

## Inherited From

* [IterativeMethodBaseModel](IterativeMethodBaseModel.md)
