# [API Reference](../../API.md) - [Others](../Others.md) - ModelParametersMerger

## Constructors

### new()

Creates a new ModelParametersMerger object. If any of the arguments are not given, default argument values for that argument will be used.

```
ModelParametersMerger.new(Model: ModelObject, modelType: string, mergeType: string): ModelParametersMergerObject
```

#### Parameters:

* Model: The model object needed to perform certain merges.

* modelType: The type where the model falls under. Available options are "regression", "classification" and "clustering".

* mergeType: Sets how a new ModelParameters is generated from given multiple ModelParameters. Available options are "average", "weightedAverage", "weightedAverageEqual", and "best".

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

* modelType: The type where the model falls under. Available options are "regression", "classification" and "clustering".

* mergeType: Sets how a new ModelParameters is generated from given multiple ModelParameters. Available options are "average", "weightedAverage", "weightedAverageEqual", and "best".

### setModelParameters()

Set the feature matrix and the label vector to perform certain merges.

```
ModelParametersMerger:setModelParametersArray(...: table/matrix)
```

#### Parameters:

* ModelParameter: A table / matrix containg the models' pararameters. The number of parameters determines number of model parameters to be used by the ModelParametersMerger.

### setData()

Set the feature matrix and the label vector to perform certain merges.

```
ModelParametersMerger:setData(featureMatrix: Matrix, labelVector: Matrix)
```

#### Parameters:

* featureMatrix: The matrix that contains all the data.

* labelVector: The matrix that contains data related to feature matrix (optional).

### generate()

Generates and returns new model parameters.

```
ModelParametersMerger:generate(): table / matrix
```

#### Returns:

* ModelParameters: The table / matrix that is generated from the merger.

## Notes

* "weightedAverage" and "best" does not work with models that do not store tables of matrices or matrix. It also does not work for sequential models. For example:

  *  DBSCAN and AffinityPropagation models.

  *  Recurrent Neural Network and LSTM models.
