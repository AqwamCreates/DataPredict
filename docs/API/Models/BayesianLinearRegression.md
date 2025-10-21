# [API Reference](../../API.md) - [Models](../Models.md) - BayesianLinearRegression

NormalLinearRegression is a supervised machine learning model that predicts continuous values (e.g. 1.2, -32, 90, -1.2 and etc. ). It uses matrix calculations to find the best model parameters.

## Stored Model Parameters

Contains a matrix.  

* ModelParameters[I][J]: Value of matrix at row I and column J. The rows are the features.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
BayesianLinearRegression.new(): ModelObject
```

#### Parameters:

* priorPrecision: The precision (inverse of variance) of the prior distribution for the model parameters. Higher values imply stronger confidence in the prior. [Default: 1]

* likelihoodPrecision: The precision of the likelihood function for the observed data. Higher values indicate more confidence in the observations. [Default: 1]

* useLogProbabilities: Set whether or not to use log probabilities when generating predicted probabilities in predict() function. [Default: False]

#### Returns:

* ModelObject: The generated model object.

## Functions

### train()

Train the model.

```
BayesianLinearRegression:train(featureMatrix: Matrix, labelVector: Matrix)
```

#### Parameters:

* featureMatrix: Matrix containing all data.

* labelVector: A (n x 1) matrix containing values related to featureMatrix.

### predict()

Predict the value for a given data.

```
BayesianLinearRegression:predict(featureMatrix: Matrix, threholdVector): number
```

#### Parameters:

* featureMatrix: Matrix containing data.

* threholdVector: A vector of thresholds for computing predicted probabilities. If provided, the model returns both predicted values and the probability that the prediction exceeds the threshold(s).

#### Returns:

* predictedValue: A value that is predicted by the model.

-- OR --

* predictedValue: A value that is predicted by the model.

* predictedProbabilityValue: The probability of the value with the given threshold.

## Inherited From

* [BaseModel](BaseModel.md)
