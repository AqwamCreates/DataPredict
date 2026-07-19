# [API Reference](../../API.md) - [Models](../Models.md) - ExtendedKalmanFilter

ExtendedKalmanFilter is a supervised machine learning model that predicts continuous values (e.g. 1.2, -32, 90, -1.2 and etc.). It uses iterative calculations to find the best model parameters.

## Stored Model Parameters

Contains a matrix.  

* ModelParameters[I][J]: Value of matrix at row I and column J. The rows are the features.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
ExtendedKalmanFilter.new(stateTransitionModelMatrix: matrix, processNoiseCovarianceMatrix: matrix, observationNoiseCovarianceMatrix: matrix, controlVector: matrix, noiseValue: number, stateFunction: function, observationStateFunction: function, stateTransitionJacobianFunction: function, observationJacobianFunction: function, lossFunction: string, useJosephForm: boolean): ModelObject
```

#### Parameters:

* stateTransitionModelMatrix: The state transition matrix that describes how the state evolves from one time step to the next without controls or noise. [Default: identity matrix]

* processNoiseCovarianceMatrix: The process noise covariance matrix representing the uncertainty in the state transition model. [Default: identity matrix]

* observationNoiseCovarianceMatrix: The observation noise covariance matrix representing the uncertainty in the observation model. [Default: identity matrix]

* controlVector: The control vector representing external inputs applied to the system. [Default: zero vector]

* noiseValue: The noise value to be used by observationNoiseCovarianceMatrix and processNoiseCovarianceMatrix. [Default: 1]

* stateFunction: The nonlinear state transition function that predicts the next state given current state and control vector. This is used when the state transition is nonlinear.

* observationStateFunction: The nonlinear observation function that maps the state space to the observation space. This is used when the observation model is nonlinear.

* stateTransitionJacobianFunction: The Jacobian matrix function of the state transition function with respect to the state. Used for linearizing the state transition during the prediction step.

* observationJacobianFunction: The Jacobian matrix function of the observation function with respect to the state. Used for linearizing the observation model during the update step.

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
ExtendedKalmanFilter:train(stateMatrix: matrix, currentStateMatrix: matrix): number[]
```

#### Parameters:

* previousStateMatrix: A matrix containing all previous state data.

* currentStateMatrix: A matrix containing all current state data.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict the value for a given data.

```
ExtendedKalmanFilter:predict(previousStateMatrix: matrix): matrix
```

#### Parameters:

* previousStateMatrix: A matrix containing all previous state data.

#### Returns:

* predictedNextStateMatrix: A vector containing next state data that are predicted by the model.

## Inherited From

* [BaseModel](BaseModel.md)
