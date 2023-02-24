# What Is Gradient Descent?

Gradient descent is one of the way on how our machine/deep learning model "learn" things. The model attempts to find the best possible solution through this technique.

# Types Of Gradient Descents

* Batch Gradient Descent: All data is used to train the model in one go.

* Mini Batch Gradient Descent: All the data is separated to multiple groups and the model will be trained based on the grouped data.

* Stochastic Gradient Descent: The model will be trained on individual data.

# Getting Started

By default, the machine/deep learning models uses batch gradient descent upon initialization. To change this we will use one of our special features.

```
local GradientDescentModes = MDLL.Others.GradientDescentModes
```

In this tutorial, we will be using "Mini Batch" gradient descent. We also need to supply the machine/deep learning model, featureMatrix and labelVector (optional). We will skip this part for now.

Then we feed it to startGradientDescentFunction(). Since we're using batch size, we ensure that the batch size value is inputted.

```
GradientDescentModes:startGradientDescent(LogisticRegressionModel, "Minibatch", featureMatrix, labelVector, 1)
```

That's all for today!
