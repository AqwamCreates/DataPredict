# [API Reference](../../API.md) - [Models](../Models.md) - BernoulliNaiveBayes

BernoulliNaiveBayes is an supervised machine learning model that predicts which classes that the input belongs to using probability.

## Stored Model Parameters

Contains a table of matrices.

* ModelParameters[1]: featureProbabilityMatrix. The columns are the features.

* ModelParameters[2]: priorProbabilityMatrix. The columns are the features.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
BernoulliNaiveBayes.new(useLogProbabilities: boolean): ModelObject
```

#### Parameters:

* useLogProbabilities: Convert the probabilities to larger values using log function.

#### Returns:

* ModelObject:  The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
BernoulliNaiveBayes:setParameters(useLogProbabilities: boolean)
```

#### Parameters:

* useLogProbabilities: Convert the probabilities to larger values using log function.

### train()

Train the model.

```
BernoulliNaiveBayes:train(featureMatrix: Matrix, labelVector: Matrix)
```

#### Parameters:

* featureMatrix: Matrix containing all data.

* labelVector: A (n x 1) matrix containing values related to featureMatrix.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict which cluster does it belong to for a given data.

```
BernoulliNaiveBayes:predict(featureMatrix: Matrix, returnOriginalOutput: boolean): Matrix, Matrix -OR- Matrix
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
BernoulliNaiveBayes:getClassesList(): []
```

#### Returns:

* ClassesList: A list of classes. The index of the class relates to which the neuron at output layer belong to. For example, {3, 1} means that the output for 3 is at first neuron, and the output for 1 is at second neuron.

### setClassesList()

```
BernoulliNaiveBayes:setClassesList(ClassesList: [])
```

#### Parameters:

* ClassesList: A list of classes. The index of the class relates to which the neuron at output layer belong to. For example, {3, 1} means that the output for 3 is at first neuron, and the output for 1 is at second neuron.

## Inherited From

* [BaseModel](BaseModel.md)
