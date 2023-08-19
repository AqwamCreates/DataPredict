# [API Reference](../../API.md) - [Models](../Models.md) - NeuralNetwork

Allows binary classification models (such as LogisticRegression) be merged together to form multi-class models.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
OneVsAll.new(maxNumberOfIterations: integer, learningRate: number, targetCost: number): ModelObject
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* targetCost: The cost at which the model stops training.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
OneVsAll:setParameters(maxNumberOfIterations: integer, learningRate: number, targetCost: number)
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* targetCost: The cost at which the model stops training.

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
NeuralNetwork:predict(featureMatrix: Matrix, returnOriginalOutput: boolean): Matrix, Matrix -OR- Matrix
```

#### Parameters:

* featureMatrix: Matrix containing all data.

* returnOriginalOutput: Set whether or not to return predicted matrix instead of value with highest probability. 

#### Returns:

* predictedVector: A vector that is predicted by the model.

* probabilityVector: A vector that contains the probability of predicted values in predictedVector.

-OR-

* predictedMatrix: A matrix containing all predicted values from all classes.

### getClassesList()

```
NeuralNetwork:getClassesList(): []
```

#### Returns:

* classesList: A list of classes. The index of the class relates to which the neuron at output layer belong to. For example, {3, 1} means that the output for 3 is at first neuron, and the output for 1 is at second neuron.

### setClassesList()

```
NeuralNetwork:setClassesList(classesList: [])
```

#### Parameters:

* classesList: A list of classes. The index of the class relates to which the neuron at output layer belong to. For example, {3, 1} means that the output for 3 is at first neuron, and the output for 1 is at second neuron.

### showDetails()

Shows the details of all layers. The details includes the number of neurons, is bias added and so on.

```
NeuralNetwork:showDetails()
```

## Inherited From

* [BaseModel](BaseModel.md)
