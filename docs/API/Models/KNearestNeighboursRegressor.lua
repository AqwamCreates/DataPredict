# [API Reference](../../API.md) - [Models](../Models.md) - KNearestNeighboursRegressor (KNN-R)

## Stored Model Parameters

Contains a table of matrices.  

* ModelParameters[1]: Feature Matrix

* ModelParameters[2]: Label Vector / Label Matrix

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
KNearestNeighboursRegressor.new(kValue: integer, distanceFunction: string, use, useWeightedDistance: boolean): ModelObject
```

#### Parameters:

* kValue: The number of closest data points taken into consideration for majority voting to determine the class of a given data point.

* distanceFunction: The distance function to be used to measure the similarity between two data points. Available options are:

  * Euclidean

  * Manhattan

  * Cosine

* useWeightedDistance: Set whether or not to use distance as a factor for prediction.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
KNearestNeighboursRegressor:setParameters(kValue: integer, distanceFunction: string, useWeightedDistance: boolean)
```

#### Parameters:

* kValue: The number of closest data points taken into consideration for majority voting to determine the class of a given data point.

* distanceFunction: The distance function to be used to measure the similarity between two data points. Available options are:

  * Euclidean

  * Manhattan

  * Cosine

### train()

Train the model.

```
KNearestNeighboursRegressor:train(featureMatrix: Matrix, labelVector: Matrix): number[]
```

#### Parameters:

* featureMatrix: Matrix containing all data.

* labelVector: A (n x 1) matrix containing values related to featureMatrix.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict the values for given data.

```
KNearestNeighboursRegressor:predict(featureMatrix: Matrix, returnOriginalOutput: boolean): Matrix, Matrix -OR- Matrix
```

#### Parameters

* featureMatrix: Matrix containing all data.

* returnOriginalOutput: Set whether or not to return predicted matrix instead of value with highest probability.

#### Returns:

* predictedlabelVector: A vector tcontaining predicted labels generated from the model.

* valueVector: A vector that contains the values of predicted labels.

-OR-

* predictedMatrix: A matrix containing all distances between stored and given data points.

## Inherited From

* [BaseModel](BaseModel.md)
