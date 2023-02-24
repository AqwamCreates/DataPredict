# API Reference - Model - NaiveBayes

NaiveBayes is an supervised machine learning model that predicts which classes that the input belongs to using probability.

## Constructors

### new()

Create new model object. If any of the arguments are not given, default argument values for that argument will be used.

```
NaiveBayes.new(): ModelObject
```

Returns:

* Model:  The generated model object.

## Functions

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

## Inherited From

* [MachineLearningBaseModel](MachineLearningBaseModel.md)

## Notes:

* Untested. May give wrong model. Use at your own risk. (I am new at understanding this model)
