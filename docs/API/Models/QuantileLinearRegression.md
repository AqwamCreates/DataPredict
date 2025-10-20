# [API Reference](../../API.md) - [Models](../Models.md) - QuantileLinearRegression

NormalLinearRegression is a supervised machine learning model that predicts continuous values (e.g. 1.2, -32, 90, -1.2 and etc. ). It uses matrix calculations to find the best model parameters.

## Stored Model Parameters

Contains a matrix.  

* ModelParameters[I][J]: Value of matrix at row I and column J. The rows are the features.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
QuantileLinearRegression.new(): ModelObject
```

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
QuantileLinearRegression:predict(featureMatrix: Matrix): number
```

#### Parameters:

* featureMatrix: Matrix containing data.

#### Returns:

* predictedValue: A value that is predicted by the model.

## Inherited From

* [BaseModel](BaseModel.md)
