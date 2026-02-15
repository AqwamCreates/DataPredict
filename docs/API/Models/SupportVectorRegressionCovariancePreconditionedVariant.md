# [API Reference](../../API.md) - [Models](../Models.md) - SupportVectorRegressionCovariancePreconditionedVariant

SupportVectorRegressionNaturalGradientVariant is a supervised machine learning model that predicts values of -1 and 1 only.

## Stored Model Parameters

Contains a matrix.  

* ModelParameters[I][J]: Value of matrix at row I and column J. The rows are the features.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
SupportVectorRegressionCovariancePreconditionedVariant.new(maximumNumberOfIterations: integer, learningRate: number, cValue: number, epsilon: number): ModelObject
```

#### Parameters:

* maximumNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between 0 to 1.

* cValue: How strict should the model can classify the data correctly. Higher the cValue, the closer the data points to the decision boundary.

* epsilon: How far the datapoint should be so that it does not contribute to the error calculation. Higher the value, the further the datapoint can be so that it does not contribute to the error calculations.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setRegularizer()

Set a regularization for the model by inputting the optimizer object.

```
SupportVectorRegressionCovariancePreconditionedVariant:setRegularizer(Regularizer: RegularizerObject)
```

#### Parameters:

* setRegularizer: The regularizer to be used.

### train()

Train the model.

```
SupportVectorRegressionCovariancePreconditionedVariant:train(featureMatrix: Matrix, labelVector: Matrix): number[]
```

#### Parameters:

* featureMatrix: Matrix containing all data.

* labelVector: A (n x 1) matrix containing values related to featureMatrix.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict the value for a given data.

```
SupportVectorRegressionCovariancePreconditionedVariant:predict(featureMatrix: Matrix): Matrix
```

#### Parameters:

* featureMatrix: Matrix containing data.

#### Returns:

* predictedVector: A vector containing values that are predicted by the model.

## Inherited From

* [GradientMethodBaseModel](GradientMethodBaseModel.md)
