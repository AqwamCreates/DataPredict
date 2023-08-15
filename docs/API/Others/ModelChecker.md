# [API Reference](../../API.md) - [Others](../Others.md) - ModelChecker

## Constructors

### new()

Creates a new ModelParametersMerger object. If any of the arguments are not given, default argument values for that argument will be used.

```
ModelChecker.new(Model: ModelObject, modelType: string, maxNumberOfIterations: integer, maxGeneralizationError: number): ModelCheckerObject
```

#### Parameters:

* Model: The model object needed to perform certain merges.

* modelType: The type where the model falls under. Available options are "regression" and "classification".

* maxNumberOfIterations: How many times should the model needed to be trained if it is being validated.

* maxGeneralizationError: The maximum generalization error so that it stops training if it is being validated. It is calculated by subtracting the validation cost and the training cost.

#### Returns:

* ModelCheckerObject: The generated ModelParametersMerger object.

## Functions

### setParameters()

Set ModelParametersMerger’s parameters. When any of the arguments are not given, previous argument values for that argument will be used.

```
ModelChecker:setParameters(Model: ModelObject, modelType: string, maxNumberOfIterations: integer, maxGeneralizationError: number)
```

#### Parameters:

* Model: The model object needed to perform certain merges.

* modelType: The type where the model falls under. Available options are "regression" and "classification".

* maxNumberOfIterations: How many times should the model needed to be trained if it is being validated.

* maxGeneralizationError: The maximum generalization error so that it stops training if it is being validated. It is calculated by subtracting the validation cost and the training cost.

### setModelParametersArray()

Set the feature matrix and the label vector to perform certain merges.

```
ModelParametersMerger:setModelParametersArray(ModelParametersArray: [])
```

#### Parameters:

* ModelParametersArray: An array containing all the model parameters from the models of same properties.

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
