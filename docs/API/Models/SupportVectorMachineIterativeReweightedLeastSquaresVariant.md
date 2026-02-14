# [API Reference](../../API.md) - [Models](../Models.md) - SupportVectorMachineIterativeReweightedLeastSquaresVariant

SupportVectorMachineIterativeReweightedLeastSquaresVariant is a supervised machine learning model that predicts values of -1 and 1 only.

## Stored Model Parameters

Contains a matrix.  

* ModelParameters[I][J]: Value of matrix at row I and column J. The rows are the features.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
SupportVectorMachineIterativeReweightedLeastSquaresVariant.new(maximumNumberOfIterations: integer, learningRate: number, cValue: number): ModelObject
```

#### Parameters:

* maximumNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between 0 to 1.

* cValue: How strict should the model can classify the data correctly. Higher the cValue, the closer the data points to the decision boundary.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setRegularizer()

Set a regularization for the model by inputting the optimizer object.

```
SupportVectorMachineIterativeReweightedLeastSquaresVariant:setRegularizer(Regularizer: RegularizerObject)
```

#### Parameters:

* setRegularizer: The regularizer to be used.

### train()

Train the model.

```
SupportVectorMachineIterativeReweightedLeastSquaresVariant:train(featureMatrix: matrix, labelVector: matrix): number[]
```

#### Parameters:

* featureMatrix: Matrix containing all data.

* labelVector: A (n x 1) matrix containing values related to featureMatrix.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict the values for given data.

```
SupportVectorMachineIterativeReweightedLeastSquaresVariant:predict(featureMatrix: Matrix, returnOriginalOutput: boolean): matrix -OR- matrix
```

#### Parameters:

* featureMatrix: Matrix containing all data.

#### Returns:

* predictedVector: A vector that is predicted by the model.

-OR-

* originalPredictedVector: A vector that contains the original predicted values.

## Inherited From

* [GradientMethodBaseModel](GradientMethodBaseModel.md)
