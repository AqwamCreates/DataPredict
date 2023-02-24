# API Reference - Model - ExpectationMaximization

ExpectationMaximization is an unsupervised machine learning model that predicts which cluster that the input belongs to using probability.

## Constructors

### new()

Create new model object. If any of the arguments are not given, default argument values for that argument will be used.

```
ExpectationMaximization.new(maxNumberOfIterations: integer, epsilon: number, numberOfClusters: integer): ModelObject
```
#### Parameters

* maxNumberOfIterations: The maximum number of iterations.

* epsilon: The target value for the model to stop training.

* numberOfClusters: Number of clusters for model to train and predict on.

#### Returns:

* Model: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are not given, previous argument values for that argument will be used.

```
ExpectationMaximization:setParameters(maxNumberOfIterations: integer, epsilon: number, numberOfClusters: integer)
```

#### Parameters

* maxNumberOfIterations: The maximum number of iterations.

* epsilon: The target value for the model to stop training.

* numberOfClusters: Number of clusters for model to train and predict on.

### train()

Train the model.

```
ExpectationMaximization:train(featureMatrix: Matrix, labelVector: Matrix)
```

#### Parameters:

* featureMatrix: Matrix containing all data.

* labelVector: A (n x 1) matrix containing values related to featureMatrix.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict which cluster does it belong to for a given data.

```
ExpectationMaximization:predict(featureMatrix: Matrix): integer, number
```

#### Parameters:

* featureMatrix: Matrix containing data.

#### Returns:

* clusterNumber: The cluster which the data belongs to.

* highestProbabilityVector: The probability (n x 1) matrix of the datapoint belongs to that particular cluster.

## Inherited From

* [MachineLearningBaseModel](MachineLearningBaseModel.md)

## Notes

* Untested. May give wrong model. Use at your own risk. (I am new at understanding this model)
