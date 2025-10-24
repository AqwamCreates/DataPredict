# [API Reference](../../API.md) - [Models](../Models.md) - BayesianLinearRegression

BayesianLinearRegression is a supervised machine learning model that predicts continuous values (e.g. 1.2, -32, 90, -1.2 and etc. ). It uses matrix calculations to find the best model parameters.

## Stored Model Parameters

Contains a matrix.  

* ModelParameters[1][I][J]: posteriorMeanVector. Value of matrix at row I and column J. The rows are the features.

* ModelParameters[2][I][J]: posteriorCovarianceMatrix. Value of matrix at row I and column J. The rows and columns are the features.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
BayesianLinearRegression.new(priorPrecision: number, likelihoodPrecision: number, useLogProbabilities: boolean): ModelObject
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
BayesianLinearRegression:predict(featureMatrix: Matrix, threholdMatrix: Matrix): Matrix -OR- Matrix, Matrix
```

#### Parameters:

* featureMatrix: Matrix containing data.

* threholdMatrix: A matrix of threshold values for computing predicted probabilities. If provided, the model returns both predicted values and the probability that the prediction exceeds the threshold(s).

#### Returns:

* predictedVector: A vector containing values that are predicted by the model.

-- OR --

* predictedVector: A vector containing values that are predicted by the model.

* predictedProbabilityMatrix: A matrix contining the probability of the values with the given threshold.

## Inherited From

* [BaseModel](BaseModel.md)
