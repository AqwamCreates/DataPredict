# [API Reference](../../API.md) - [Others](../Others.md) - GradientDescentModes

## Functions

### startGradientDescent()

Runs the machine/deep learning model specific gradient descent mode.

```
GradientDescentMode:startGradientDescent(Model: ModelObject, mode: string, featureMatrix: Matrix, labelVector: Matrix, numberOfBatches: integer)
```

#### Parameters:

* Model: The model that you want to train

* mode: The mode of gradient descent. Available modes are "Batch", "Minibatch" and "Stochastic".

* featureMatrix: Matrix containing all data.

* labelVector: A (n x 1) matrix containing values related to featureMatrix.

* numberOfBatches: Number of batches for the model.

#### Note:

* When using "Stochastic" mode, make sure you set the model's max number of iterations to 1.

* numberOfBatches is only applicable for "Minibatch" mode.
