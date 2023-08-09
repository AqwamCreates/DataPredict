# [API Reference](../../API.md) - [Others](../Others.md) - GradientDescentModes

## Functions

### startGradientDescent()

Runs the machine/deep learning model specific gradient descent mode.

```
GradientDescentMode:startGradientDescent(Model: ModelObject, mode: string, featureMatrix: Matrix, labelVector: Matrix, batchSize: integer, showOutputCost: boolean)
```

#### Parameters:

* Model: The model that you want to train

* mode: The mode of gradient descent. Available modes are "Batch", "MiniBatch" and "Stochastic".

* featureMatrix: Matrix containing all data.

* labelVector: A (n x 1) matrix containing values related to featureMatrix.

* batchSize: The batch size to be inputted into the model.

* showOutputCost: Set whether or not to show the final cost for each epoch (MiniBatch) or data (Stochastic).

#### Note:

* When using "Stochastic" mode, make sure you set the model's max number of iterations to 1.

* numberOfBatches is only applicable for "Minibatch" mode.
