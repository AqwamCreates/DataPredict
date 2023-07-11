# [API Reference](../../API.md) - [Models](../Models.md) - NaiveBayes

NaiveBayes is an supervised machine learning model that predicts which classes that the input belongs to using probability.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
NaiveBayes.new(useLogProbabilities: boolean): ModelObject
```

#### Parameters:

* useLogProbabilities: Convert the probabilities to larger values using log function.

#### Returns:

* Model:  The generated model object.

## Functions

### setParameters()

Set the parameters for the model

```
NaiveBayes:setParameters(useLogProbabilities: boolean)
```

#### Parameters:

* useLogProbabilities: Convert the probabilities to larger values using log function.

### train()

Train the model.

```
NaiveBayes:train(featureMatrix: Matrix, labelVector: Matrix)
```

#### Parameters:

* featureMatrix: Matrix containing all data.

* labelVector: A (n x 1) matrix containing values related to featureMatrix.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict which cluster does it belong to for a given data.

```
NaiveBayes:predict(featureMatrix: Matrix): integer, number
```

#### Parameters:

* featureMatrix: Matrix containing data.

#### Returns:

* clusterNumber: The cluster which the data belongs to.

* highestProbabilityVector: The probability (n x 1) matrix of the datapoint belongs to that particular cluster.

### getClassesList()

```
NaiveBayes:getClassesList(): []
```

#### Returns:

* classesList: A list of classes. The index of the class relates to which the neuron at output layer belong to. For example, {3, 1} means that the output for 3 is at first neuron, and the output for 1 is at second neuron.

### setClassesList()

```
NaiveBayes:setClassesList(classesList: [])
```

#### Parameters:

* classesList: A list of classes. The index of the class relates to which the neuron at output layer belong to. For example, {3, 1} means that the output for 3 is at first neuron, and the output for 1 is at second neuron.

### showDetails()

Shows the details of all layers. The details includes the number of neurons, is bias added and so on.


## Inherited From

* [BaseModel](BaseModel.md)

## Notes

* Untested. May give wrong model. Use at your own risk. (I am new at understanding this model)
