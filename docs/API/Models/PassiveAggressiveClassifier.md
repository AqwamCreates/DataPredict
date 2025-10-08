# [API Reference](../../API.md) - [Models](../Models.md) - PassiveAggressiveClassifier

PassiveAggressiveClasifier is a supervised machine learning model that predicts binary values (+1 and -1). It uses iterative calculations to find the best model parameters.

## Stored Model Parameters

Contains a matrix.  

* ModelParameters[I][J]: Value of matrix at row I and column J. The rows are the features.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
PassiveAggressiveClassifier.new(maximumNumberOfIterations: integer, variant: string, cValue: number): ModelObject
```

#### Parameters:

* maximumNumberOfIterations: How many times should the model needed to be trained. [Default: math.huge]

* variant: Controls which PassiveAggressiveClasifier variant to use. Available options are:

    * 0 (Default)
 
    * 1
 
    * 2

* cValue: The aggressiveness parameter used in some Passive-Aggressive variants (1 and 2). Higher values allow larger updates per misclassified example, while lower values limit the update magnitude. [Default: 1]

#### Returns:

* ModelObject: The generated model object.

## Functions

### train()

Train the model.

```
PassiveAggressiveClassifier:train(featureMatrix: Matrix, labelVector: Matrix): number[]
```

#### Parameters:

* featureMatrix: Matrix containing all data.

* labelVector: A (n x 1) matrix containing values related to featureMatrix.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict the values for given data.

```
PassiveAggressiveClasifier:predict(featureMatrix: Matrix, returnOriginalOutput: boolean): Matrix -OR- Matrix
```

#### Parameters:

* featureMatrix: Matrix containing data.

* returnOriginalOutput: Set whether or not to return original output.

#### Returns:

* predictedLabelVector: A vector containing the classes that is predicted by the model.

-OR-

* predictedValueVector: A vector containing the values that is predicted by the model.

## Inherited From

* [IterativeMethodBaseModel](IterativeMethodBaseModel.md)
