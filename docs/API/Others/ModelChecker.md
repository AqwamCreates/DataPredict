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

### setClassesList()

Set the feature matrix and the label vector to perform certain merges.

```
ModelChecker:setClassesList(classesList: [])
```

#### Parameters:

* classesList: A list of classes. The index of the class relates to which the neuron at output layer belong to. For example, {3, 1} means that the output for 3 is at first neuron, and the output for 1 is at second neuron.

### test()

Test the model.

```
ModelChecker:test(testFeatureMatrix: Matrix, testLabelVector: Matrix): number, Matrix, Matrix
```

#### Parameters:

* testFeatureMatrix: Matrix containing all data.

* testLabelVector: A (n x 1) matrix containing values related to featureMatrix.

#### Returns:

* testCost

* errorVector

* predictedLabelMatrix

### validate()

Validate the model.

```
ModelChecker:validate(trainFeatureMatrix: Matrix, trainLabelVector: Matrix, validationFeatureMatrix: Matrix, validationLabelVector: Matrix): number[], number[]
```

#### Parameters:

* featureMatrix: Matrix containing all data.

* labelVector: A (n x 1) matrix containing values related to featureMatrix.

#### Returns:

* trainCostArray

* validationCostArray
