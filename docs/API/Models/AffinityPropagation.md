# [API Reference](../../API.md) - [Models](../Models.md) - AffinityPropagation

AffinityPropagation is an unsupervised machine learning model that predicts which cluster that the input belongs to using a similarity matrix that measures the similarity between each data point.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
AffinityPropagation.new(maxNumberOfIterations: integer, damping: number, targetCost: integer): ModelObject
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* damping: A high value leads to fewer changes, while a low value leads to more exploration. The value can be set between 0 and 1.

* targetCost: The cost at which the model stops training.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
AffinityPropagation:setParameters(maxNumberOfIterations: integer, damping: number, targetCost: integer)
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* damping: A high value leads to fewer changes, while a low value leads to more exploration. The value is set between 0 and 1.

* targetCost: The cost at which the model stops training.

### canAppendPreviousFeatureMatrix()

Set the option if the previous feature matrix should append with the new feature matrix during training.

```
AffinityPropagationModel:canAppendPreviousFeatureMatrix(option: boolean)
```

#### Parameters:

* option: the boolean value to set whether or not the feature matrices can be appended.

### clearPreviousFeatureMatrix()

Clears previous feature matrix that was stored in the model.

```
AffinityPropagationModel:clearPreviousFeatureMatrix()
```

### getPreviousFeatureMatrix()

Get the previous feature matrix that was stored in the model.

```
AffinityPropagationModel:getPreviousFeatureMatrix(): matrix
```

### returns:

* featureMatrix: a matrix containing data that was previously stored in the model

### train()

Train the model.

```
AffinityPropagation:train(featureMatrix: Matrix)
```

#### Parameters:

* featureMatrix: Matrix containing all data.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict which cluster does it belong to for a given data.

```
AffinityPropagation:predict(featureMatrix: Matrix): integer, number
```

#### Parameters:

* featureMatrix: Matrix containing data.

#### Returns:

* clusterNumber: The cluster which the data belongs to.

* shortestDistance: The distance between the datapoint and the center of the cluster (centroids).

## Inherited From

* [BaseModel](BaseModel.md)
