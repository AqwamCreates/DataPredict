# [API Reference](../../API.md) - [Models](../Models.md) - UnscentedKalmanFilter

UnscentedKalmanFilter is a supervised machine learning model that predicts continuous values (e.g. 1.2, -32, 90, -1.2 and etc.). It uses iterative calculations to find the best model parameters.

## Stored Model Parameters

Contains a matrix.  

* ModelParameters[I][J]: Value of matrix at row I and column J. The rows are the features.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
UnscentedKalmanFilter.new(alpha: number, beta: number: kappa: number, noiseValue: number, lossFunction: string, useJosephForm: boolean, epsilon: number, processNoiseCovarianceMatrix: matrix, observationNoiseCovarianceMatrix: matrix): ModelObject
```

#### Parameters:

* alpha: The spread of the sigma points around the mean state. Typically set to a small positive value. Controls the distribution of the sigma points. [Default: 1e-3]

* beta: Used to incorporate prior knowledge of the distribution. For Gaussian distributions, beta = 2 is optimal. [Default: 2]

* kappa: A secondary scaling parameter that determines the distance of the sigma points from the mean. Usually set to 0 or 3 - n, where n is the state dimension. [Default: 0]

* noiseValue: The noise value to be used by observationNoiseCovarianceMatrix and processNoiseCovarianceMatrix. [Default: 1]

* lossFunction: The function to calculate the cost of each training. Available options are:

  * L1

  * L2 (Default)

  * Mahalanobis

* useJosephForm: Set whether or not to use a more numerically accurate representation in exchange for lower performance [Default: true]

* epsilon: Used for Jacobian approximation. [Default: 1e-5]

* stateTransitionFunction: The non-linear state transition function that predicts the next state given current state and control vector. This function defines how the system evolves over time without noise. [Default: linear state transition]

* observationFunction: The nonlinear observation function that maps the state space to the observation space. This function defines how measurements relate to the state. [Default: linear observation]

* processNoiseCovarianceMatrix: The process noise covariance matrix representing the uncertainty in the state transition model. [Default: identity matrix]

* observationNoiseCovarianceMatrix: The observation noise covariance matrix representing the uncertainty in the observation model. [Default: identity matrix]

#### Returns:

* ModelObject: The generated model object.

## Functions

### train()

Train the model.

```
UnscentedKalmanFilter:train(stateMatrix: matrix, labelVector: matrix): number[]
```

#### Parameters:

* stateMatrix: Matrix containing data.

* labelVector: A (n x 1) matrix containing values related to featureMatrix.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict the value for a given data.

```
UnscentedKalmanFilter:predict(stateMatrix: matrix): matrix
```

#### Parameters:

* stateMatrix: Matrix containing data.

#### Returns:

* predictedVector: A vector containing values that are predicted by the model.

## Inherited From

* [BaseModel](BaseModel.md)
