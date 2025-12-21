# [API Reference](../../API.md) - [Models](../Models.md) - OrdinalRegression

## Stored Model Parameters

Contains a table of matrices.  

* ModelParameters[1]: weightMatrix

* ModelParameters[2]: thresholdVector

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
OrdinalRegression.new(binaryFunction: string, use, useWeightedDistance: boolean): ModelObject
```

#### Parameters:

* binaryFunction

#### Returns:

* ModelObject: The generated model object.

## Functions

### train()

Train the model.

```
OrdinalRegression:train(featureMatrix: Matrix, labelVector: Matrix): number[]
```

#### Parameters:

* featureMatrix: Matrix containing all data.

* labelVector: A (n x 1) matrix containing values related to featureMatrix.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict the values for given data.

```
OrdinalRegression:predict(featureMatrix: Matrix, returnOriginalOutput: boolean): Matrix, Matrix -OR- Matrix
```

#### Parameters

* featureMatrix: Matrix containing all data.

* returnOriginalOutput: Set whether or not to return predicted matrix instead of value with highest probability.

#### Returns:

* predictedlabelVector: A vector containing predicted labels generated from the model.

* valueVector: A vector that contains the values of predicted labels.

-OR-

* predictedMatrix: A matrix containing all the probabilities.

## Inherited From

* [GradientMethodBaseModel](GradientMethodBaseModel.md)
