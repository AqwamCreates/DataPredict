# [API Reference](../../API.md) - [Others](../Others.md) - GradientDescentModifier

## Constructors

### new()

Creates a new gradient descent modifier object. If any of the arguments are not given, default argument values for that argument will be used.

```
GradientDescentModifier.new(Model: ModelObject, gradientDescentType: string, batchSize: integer, showOutput: boolean): GradientDescentModifierObject
```

#### Parameters:

* Model: The model object to modify its gradient descent capabilities.

* gradientDescentType: The type of gradient descent to be used when train() function is called. Available modes are "Batch", "MiniBatch" and "Stochastic".

* batchSize: The batch size to split the featureMatirx and labelVector into multiple parts.

* showOutput: Set whether or not to show the final cost for each epoch (MiniBatch) or data (Stochastic).

#### Returns:

* GradientDescentModifierObject: A gradient descent modifier object that uses the model's train() and predict() functions so that it behaves like a regular model.

## Functions

### setParameters()

Set modifier’s parameters. When any of the arguments are not given, previous argument values for that argument will be used.

```
GradientDescentModifier:setParameters(Model: ModelObject, gradientDescentType: string, batchSize: integer, showOutput: boolean)
```

#### Parameters:

* Model: The model object to modify its gradient descent capabilities.

* gradientDescentType: The type of gradient descent to be used when train() function is called. Available modes are "Batch", "MiniBatch" and "Stochastic".

* batchSize: The batch size to split the featureMatirx and labelVector into multiple parts.

* showOutput: Set whether or not to show the final cost for each epoch (MiniBatch) or data (Stochastic).

### train()

Trains the machine/deep learning model under specific gradient descent mode.

```
GradientDescentModifier:train(...): number[]
```

#### Parameters:

* ...: The parameters are the same to the original model's train() function.

#### Returns:

* costArray: An array containing cost values.

### predict()

* Predict the value for a given data.

```
GradientDescentModifier:predict(...): ...
```

#### Parameters:

...: The parameters are the same to the original model's predict() function.

#### Returns:

...: The outputs are the same to the original model's predict() function.

## Note:

* When using "Stochastic" mode, make sure you set the model's max number of iterations to 1.

* numberOfBatches is only applicable for "Minibatch" mode.
