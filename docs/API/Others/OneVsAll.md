# [API Reference](../../API.md) - [Models](../Models.md) - OneVsAll

Allows binary classification models (such as LogisticRegression) be merged together to form multi-class models.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
OneVsAll.new(maxNumberOfIterations: integer, useNegativeOneBinaryLabel: boolean, targetCost: number): OneVsAllObject
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* useNegativeOneBinaryLabel: Set whether or not if the incorrect label uses -1 instead of 0

* targetCost: The cost at which the model stops training.

#### Returns:

* OneVsAllObject: The generated OneVsAll object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
OneVsAll:setParameters(maxNumberOfIterations: integer, learningRate: number, targetCost: number)
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* useNegativeOneBinaryLabel: Set whether or not if the incorrect label uses -1 instead of 0

* targetCost: The cost at which the model stops training.

### setModels()

```
OneVsAll:setModels(...: ModelObject)
```

#### Parameters:

* ...: Models to be added. Each model correspond to each class.

### setParameters()

```
OneVsAll:setAllModelsParameters(...: any)
```

#### Parameters:

* ...: The parameters to be set to all models stored in this OneVsAll object. Not to be confused with ModelParameters that stores table of matrices or matrix.

### train()

Train the model.

```
NeuralNetwork:train(featureMatrix: Matrix, labelVector / labelMatrix: Matrix): number[]
```
#### Parameters:

* featureMatrix: Matrix containing all data.

* labelVector / labelMatrix: A (n x 1) / (n x o) matrix containing values related to featureMatrix. When using the label matrix, the number of columns must be equal to number of classes.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict the values for given data.

```
OneVsAll:predict(featureMatrix: Matrix): Matrix, Matrix
```

#### Parameters:

* featureMatrix: Matrix containing all data.

* returnOriginalOutput: Set whether or not to return predicted matrix instead of value with highest probability. 

#### Returns:

* predictedVector: A vector that is predicted by the model.

* highestValueVector: A vector that contains the predicted values in predictedVector.

### getClassesList()

```
OneVsAll:getClassesList(): []
```

#### Returns:

* classesList: A list of classes. The index of the list relates to which model belong to. For example, {3, 1} means that the output for 3 is at first model, and the output for 1 is at second model.

### setClassesList()

```
OneVsAll:setClassesList(classesList: [])
```

#### Parameters:

* classesList: A list of classes. The index of the list relates to which model belong to. For example, {3, 1} means that the output for 3 is at first model, and the output for 1 is at second model.

## Inherited From

* [BaseModel](BaseModel.md)
