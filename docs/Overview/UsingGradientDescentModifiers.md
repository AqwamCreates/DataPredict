# What Is Gradient Descent?

Gradient descent is one of the way on how our machine/deep learning model "learn" things. The model attempts to find the best possible solution through this technique.

# Types Of Gradient Descents

* Batch Gradient Descent: All data is used to train the model in one go.

* Mini Batch Gradient Descent: All the data is separated to multiple groups and the model will be trained based on the grouped data.

* Stochastic Gradient Descent: The model will be trained on individual data.

# Getting Started

By default, the machine/deep learning models uses batch gradient descent upon initialization. To change this we create a GradientDescentModifier object.

We will modify the graident descent to mini batch gradient descent.

```lua
local Model = MDLL.Models.SupportVectorMachine.new()

local ModifiedModel = MDLL.Others.GradientDescentModifier.new(SupportVectorMachine, "MiniBatch")
```

Once that is set up, you can call train() and predict() functions from the ModifiedModel. This is because it uses the original model's train() and predict() functions.

In other words, you can do what the original model can do, except the behaviour of the gradient descent and the cost has been changed.

```lua
local costArray = ModifiedModel:train(featureMatrix, labelVector)

local predictedVector = ModifiedModel:predict(featureMatrix2)
```

Looks pretty similar huh? You can try to combine this with other functionalitiees as well.

That's all for today!
