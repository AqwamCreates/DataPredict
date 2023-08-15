# [API Reference](../../API.md) - [Others](../Others.md) - ModelChecker

## Functions

## testRegressionModel()

```
ModelChecking:testRegressionModel(Model: ModelObject, featureMatrix: Matrix, labelVector: Matrix): number, matrix, matrix
```

#### Parameters:

* Model: The model you want to test.

* featureMatrix: Matrix containing all data.

* labelVector: A (n x 1) matrix containing values related to featureMatrix.

#### Returns:

* averageError: The average error made by the model.

* errorVector: A (n x 1) matrix containing the error values relative to the featureMatrix position.

* modelOutputVector: A (n x 1) matrix containing the predicted values from the model relative to the featureMatrix position.

## testClassificationModel()

```
ModelChecking:testRegressionModel(Model: ModelObject, featureMatrix: Matrix, labelVector: Matrix): number, number[], number[], matrix
```

#### Parameters:

* Model: The model you want to test.

* featureMatrix: Matrix containing all data.

* labelVector: A (n x 1) matrix containing values related to featureMatrix.

#### Returns:

* averageError: The average error made by the model.

* correctAtDataArray: An array containing all the positions that are correctly predicted by the model.

* wrongAtDataArray: An array containing all the positions that are incorrectly predicted by the model.

* modelOutputVector: A (n x 1) matrix containing the predicted values from the model relative to the featureMatrix position.
