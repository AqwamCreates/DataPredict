# [API Reference](../../API.md) - [Others](../Others.md) - GradientDescentModifier

## Constructors

### new()

Creates a new gradient descent modifier object. If any of the arguments are not given, default argument values for that argument will be used.

```
GradientDescentModifier.new(Model: ModelObject, gradientDescentType: string, batchSize: integer, showOutput: boolean): RegularizationObject
```

#### Parameters:

* Model: The model object to modify its gradient descent capabilities.

* gradientDescentType: The type of gradient descent to be used when train() function is called. Available modes are "Batch", "MiniBatch" and "Stochastic".

* batchSize: The batch size to be inputted into the model.

* showOutput: Set whether or not to show the final cost for each epoch (MiniBatch) or data (Stochastic).

## Functions

### setParameters()

Set modifier’s parameters. When any of the arguments are not given, previous argument values for that argument will be used.

```
GradientDescentModifier:setParameters(Model: ModelObject, gradientDescentType: string, batchSize: integer, showOutput: boolean)
```

#### Parameters:

* Model: The model object to modify its gradient descent capabilities.

* gradientDescentType: The type of gradient descent to be used when train() function is called. Available modes are "Batch", "MiniBatch" and "Stochastic".

* batchSize: The batch size to be inputted into the model.

* showOutput: Set whether or not to show the final cost for each epoch (MiniBatch) or data (Stochastic).

### train()

Trains the machine/deep learning model under specific gradient descent mode.

```
GradientDescentModifier:train(featureMatrix, labelVector): number[]
```

#### Parameters:

* featureMatrix: Matrix containing all data.

* labelVector: A (n x 1) matrix containing values related to featureMatrix.

#### Returns:

* costArray: An array containing cost values.

### predict()

* Predict the value for a given data.

```
GradientDescentModifier:predict(featureMatrix: Matrix, returnOriginalOutput: boolean): integer, number -OR- Matrix
```

#### Parameters:

* featureMatrix: Matrix containing all data.

* returnOriginalOutput: Set whether or not to return predicted matrix instead of value with highest probability. 

#### Returns:

* predictedValue: A value that is predicted by the model.

* probability: The probability of predicted value.

-OR-

* predictedMatrix: A matrix containing all predicted values from all classes.

## Note:

* When using "Stochastic" mode, make sure you set the model's max number of iterations to 1.

* numberOfBatches is only applicable for "Minibatch" mode.
