# [API Reference](../../API.md) - [Others](../Others.md) - ModelParametersMerger

It is a class for handling the merging of model parameters.

## Notes

* "WeightedAverage" and "Best" does not work with models that do not store tables of matrices or matrix. It also does not work for models with sequential output. For example:

  *  DBSCAN and AffinityPropagation models. (Contains non-matrices)

  *  Recurrent Neural Network and LSTM models. (The outputs are sequential)

## Constructors

### new()

Creates a new ModelParametersMerger object. If any of the arguments are not given, default argument values for that argument will be used.

```
ModelParametersMerger.new(Model: ModelObject, modelType: string, mergeType: string): ModelParametersMergerObject
```

#### Parameters:

* Model: The model object needed to perform certain merges.

* modelType: The type where the model falls under. Available options are "Regression", "Classification" and "Clustering".

* mergeType: Sets how a new ModelParameters is generated from given multiple ModelParameters. Available options are:

  * Average

  * WeightedAverage

  * WeightedAverageEqual

  * Best

  * Custom

#### Returns:

* ModelParametersMergerObject: The generated ModelParametersMerger object.

## Functions

### setParameters()

Set ModelParametersMerger’s parameters. When any of the arguments are not given, previous argument values for that argument will be used.

```
ModelParametersMerger:setParameters(Model: ModelObject, modelType: string, mergeType: string)
```

#### Parameters:

* Model: The model object needed to perform certain merges.

* modelType: The type where the model falls under. Available options are "Regression", "Classification" and "Clustering".

* mergeType: Sets how a new ModelParameters is generated from given multiple ModelParameters. Available options are:

  * Average

  * WeightedAverage

  * WeightedAverageEqual

  * Best

  * Custom

### merge()

Merges existing model parameters to create a new one.

```
ModelParametersMerger:merge(...: table/matrix): table / matrix
```

#### Parameters:

* ModelParameter: A table / matrix containg the models' pararameters. The number of parameters determines number of model parameters to be used by the ModelParametersMerger.

#### Returns:

* ModelParameters: The table / matrix that is generated from the merger.

### setCustomSplitPercentageArray()

Sets a custom split percentage for each of the model parameters. This will be used when "custom" merge type is used.

```
ModelParametersMerger:setCustomSplitPercentageArray(splitPercentageArray: number[])
```

#### Parameters:

* Sets the percentage to be applied to individual model parameters. The index of each value in the array determines which model parameter it applies to. For example, if you place 0.5 in the first index, it will apply a 0.5 multiplier to the first model parameter. It is recommended that the sum of all values in the array equals 1.

### setData()

Set the feature matrix and the label vector to perform certain merges.

```
ModelParametersMerger:setData(featureMatrix: Matrix, labelVector: Matrix)
```

#### Parameters:

* featureMatrix: The matrix that contains all the data.

* labelVector: The matrix that contains data related to feature matrix (optional).

## Inherited From

* [BaseInstance](../Cores/BaseInstance.md)