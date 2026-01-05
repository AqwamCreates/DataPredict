# [API Reference](../../API.md) - [Models](../Models.md) - LocalOutlierFactor

## Stored Model Parameters

Contains a table of matrices.  

* ModelParameters: Feature Matrix

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
LocalOutlierFactor.new(kValue: integer, distanceFunction: string, use, useWeightedDistance: boolean): ModelObject
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
LocalOutlierFactor:train(featureMatrix: matrix): number[]
```

#### Parameters:

* featureMatrix: Matrix containing all data.

### score()

Generates the score vector.

```
LocalOutlierFactor:score(): matrix
```

#### Returns:

* scoreVector: A vector containing the scores for each data stored in train() function. The higher the value, the higher it is a normal datapoint.

## Inherited From

* [BaseModel](BaseModel.md)

## References

* [LOF: identifying density-based local outliers](https://dl.acm.org/doi/10.1145/335191.335388)
