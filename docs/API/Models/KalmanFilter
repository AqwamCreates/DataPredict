# [API Reference](../../API.md) - [Models](../Models.md) - KalmanFilter

KalmanFilter is a supervised machine learning model that predicts continuous values (e.g. 1.2, -32, 90, -1.2 and etc.). It uses iterative calculations to find the best model parameters.

## Stored Model Parameters

Contains a matrix.  

* ModelParameters[I][J]: Value of matrix at row I and column J. The rows are the features.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
KalmanFilter.new(stateTransitionModelMatrix: matrix, observationModelMatrix: matrix, processNoiseCovarianceMatrix: matrix, observationNoiseCovarianceMatrix: matrix, controlInputMatrix: matrix, controlVector: matrix, noiseValue: number, lossFunction: string, useJosephForm: boolean): ModelObject
```

#### Parameters:

* stateTransitionModelMatrix: How many times should the model needed to be trained.

* observationModelMatrix: The speed at which the model learns. Recommended that the value is set between 0 to 1.

* processNoiseCovarianceMatrix: The function to calculate the cost of each training. Available options are:

* observationNoiseCovarianceMatrix:

* controlInputMatrix

* controlVector

* noiseValue: [Default: 1]

* lossFunction: The function to calculate the cost of each training. Available options are:

  * L1

  * L2 (Default)

  * Mahalanobis

* useJosephForm: Set whether or not to use a more numerically accurate representation in exchange for lower performance [Default: true]

#### Returns:

* ModelObject: The generated model object.

## Functions

### train()

Train the model.

```
KalmanFilter:train(featureMatrix: Matrix, labelVector: Matrix): number[]
```

#### Parameters:

* featureMatrix: Matrix containing all data.

* labelVector: A (n x 1) matrix containing values related to featureMatrix.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict the value for a given data.

```
KalmanFilter:predict(featureMatrix: Matrix): Matrix
```

#### Parameters:

* featureMatrix: Matrix containing data.

#### Returns:

* predictedVector: A vector containing values that are predicted by the model.

## Inherited From

* [BaseModel](BaseModel.md)
