# [API Reference](../../API.md) - [Models](../Models.md) - BayesianQuantileLinearRegression

BayesianQuantileLinearRegression is a supervised machine learning model that predicts continuous values (e.g. 1.2, -32, 90, -1.2 and etc. ). It uses matrix calculations to find the best model parameters.

## Stored Model Parameters

Contains a matrix.  

* ModelParameters[1][I][J]: posteriorMeanVector. Value of matrix at row I and column J. The rows are the features.

* ModelParameters[2][I][J]: posteriorCovarianceMatrix. Value of matrix at row I and column J. The rows and columns are the features.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
BayesianQuantileLinearRegression.new(priorPrecision: number, likelihoodPrecision: number): ModelObject
```

#### Parameters:

* priorPrecision: How confident we are with the previous model weights. Higher values means more confidence in the previous model weights. [Default: 1]

* likelihoodPrecision: How reliable are the observations. Higher values means observations are more reliable. [Default: 1]

#### Returns:

* ModelObject: The generated model object.

## Functions

### train()

Train the model.

```
BayesianQuantileLinearRegression:train(featureMatrix: Matrix, labelVector: Matrix)
```

#### Parameters:

* featureMatrix: Matrix containing all data.

* labelVector: A (n x 1) matrix containing values related to featureMatrix.

### predict()

Predict the value for a given data.

```
BayesianQuantileLinearRegression:predict(featureMatrix: Matrix, quantileMatrix: Matrix): Matrix -OR- Matrix, Matrix
```

#### Parameters:

* featureMatrix: Matrix containing data.

* quantileMatrix: A matrix of quantile values to predict for each data point. Must be between 0 and 1.

#### Returns:

* predictedVector: A vector containing values that are predicted by the model.

-- OR --

* predictedVector: A vector containing values that are predicted by the model.

* predictedQuantileMatrix: A matrix containing the predicted values corresponding to each quantile in quantileMatrix.

## Inherited From

* [BaseModel](BaseModel.md)
