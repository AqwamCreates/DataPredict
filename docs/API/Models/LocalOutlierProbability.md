# [API Reference](../../API.md) - [Models](../Models.md) - LocalOutlierProbability

## Stored Model Parameters

Contains a table of matrices.  

* ModelParameters: Feature Matrix

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
LocalOutlierProbability.new(kValue: integer, distanceFunction: string, use, useWeightedDistance: boolean): ModelObject
```

#### Parameters:

* kValue: The number of closest data points taken into consideration for majority voting to determine the class of a given data point.

* distanceFunction: The distance function to be used to measure the similarity between two data points. Available options are:

  * Euclidean

  * Manhattan

  * Cosine

#### Returns:

* ModelObject: The generated model object.

## Functions

### train()

Train the model.

```
LocalOutlierProbability:train(featureMatrix: matrix): number[]
```

#### Parameters:

* featureMatrix: Matrix containing all data.

### score()

Generates the score vector.

```
LocalOutlierProbability:score(): matrix
```

#### Returns:

* scoreVector: A vector containing the scores for each data stored in train() function.

## Inherited From

* [BaseModel](BaseModel.md)
