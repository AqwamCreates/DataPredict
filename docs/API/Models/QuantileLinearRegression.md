# [API Reference](../../API.md) - [Models](../Models.md) - QuantileLinearRegression

QuantileLinearRegression is a supervised machine learning model that predicts continuous values (e.g. 1.2, -32, 90, -1.2 and etc. ). It uses matrix calculations to find the best model parameters.

## Stored Model Parameters

Contains a matrix.  

* ModelParameters[1][I][J]: posteriorMeanVector. Value of matrix at row I and column J. The rows are the features.

* ModelParameters[2][I][J]: posteriorCovarianceMatrix. Value of matrix at row I and column J. The rows and columns are the features.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
QuantileLinearRegression.new(priorPrecision: number, likelihoodPrecision: number, useLogProbabilities: boolean): ModelObject
```

#### Parameters:

* priorPrecision: The precision (inverse of variance) of the prior distribution for the model parameters. Higher values imply stronger confidence in the prior. [Default: 1]

* likelihoodPrecision: The precision of the likelihood function for the observed data. Higher values indicate more confidence in the observations. [Default: 1]

#### Returns:

* ModelObject: The generated model object.

## Functions

### train()

Train the model.

```
QuantileLinearRegression:train(featureMatrix: Matrix, labelVector: Matrix)
```

#### Parameters:

* featureMatrix: Matrix containing all data.

* labelVector: A (n x 1) matrix containing values related to featureMatrix.

### predict()

Predict the value for a given data.

```
QuantileLinearRegression:predict(featureMatrix: Matrix, quantileVector: Matrix): Matrix -OR- Matrix, Matrix
```

#### Parameters:

* featureMatrix: Matrix containing data.

* quantileVector: A matrix or vector specifying the quantiles to predict for each data point. Must be between 0 and 1.

#### Returns:

* predictedVector: A vector containing values that are predicted by the model.

-- OR --

* predictedVector: A vector containing values that are predicted by the model.

* predictedQuantileVector: A vector containing the predicted values corresponding to each quantile in quantileVector.

## Inherited From

* [BaseModel](BaseModel.md)
