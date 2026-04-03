# [API Reference](../../API.md) - [Models](../Models.md) - TheilSenRegression

TheilSenRegression is a supervised machine learning model that predicts continuous values (e.g. 1.2, -32, 90, -1.2 and etc. ). It uses matrix calculations to find the best model parameters.

## Stored Model Parameters

Contains a matrix.  

* ModelParameters[1][1]: medianSlopeValue.

* ModelParameters[1][2]: medianBiasValue.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
TheilSenRegression.new(): ModelObject
```

#### Returns:

* ModelObject: The generated model object.

## Functions

### train()

Train the model.

```
TheilSenRegression:train(featureMatrix: Matrix, labelVector: Matrix)
```

#### Parameters:

* featureMatrix: Matrix containing all data.

* labelVector: A (n x 1) matrix containing values related to featureMatrix.

### predict()

Predict the value for a given data.

```
TheilSenRegression:predict(featureMatrix: Matrix): Matrix
```

#### Parameters:

* featureMatrix: Matrix containing data.

#### Returns:

* predictedVector: A vector containing values that are predicted by the model.

## Inherited From

* [BaseModel](BaseModel.md)
