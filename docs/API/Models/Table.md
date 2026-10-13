# [API Reference](../../API.md) - [Models](../Models.md) - Table

## Constructors

### new()

Creates a new base model object. If any of the arguments are nil, default argument values for that argument will be used.

```
Table.new(learningRate: number, Optimizer: Optimizer: Object, FeaturesList: {any}, ClassesList: {any}): ModelObject
```

#### Parameters:

* learningRate: The speed at which the algorithm learns. Recommended to set between 0 and 1.

* Optimizer: The optimizer object to be used.

* FeaturesList: A list containing all the features.

* ClassesList: A list containing all the classes.

* costFunction: The function to calculate the cost of each training. Available options are:

| Function                   | Input Range (From Final Activation Layer And Label Vector) | Output Range | Characteristics                                  | Use Cases                                                              |
|----------------------------|------------------------------------------------------------|--------------|--------------------------------------------------|------------------------------------------------------------------------|
| MeanSquaredError (Default) | (-∞, ∞)                                                    | [0, ∞)       | Quadratic penalty, sensitive to outliers         | Regression with normally distributed errors, when outliers are minimal |
| MeanAbsoluteError          | (-∞, ∞)                                                    | [0, ∞)       | Linear penalty, robust to outliers               | Robust regression, when dataset contains outliers, median prediction   |
| HingeLoss                  | (-∞, ∞)                                                    | (0, ∞)       | Creates max-margin decision boundary             | Maximum margin classification, text categorization                     |
| SquaredHingeLoss           | (-∞, ∞)                                                    | (0, ∞)       | Creates sensitive max-margin decision boundary   | Sensitive maximum margin classification, text categorization           |
| PerceptronLoss             | (-∞, ∞)                                                    | (0, ∞)       | Adjusts the boundary when incorrectly classified | Margin classification, text categorization                             |
| BinaryCrossEntropy         | (0, 1)                                                     | (0, ∞)       | Penalizes confident wrong predictions heavily    | Probabilistic-based classification                                     |

#### Returns:

* ModelObject: The generated model object.

## Functions

### train()

```
Table:predict(featureVector, labelVector)
```

### predict()

```
Table:predict(featureVector, returnOriginalOutput)
```

## Inherited From

* [BaseModel](BaseModel.md)
