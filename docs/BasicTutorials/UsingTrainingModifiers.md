# Types Of Training Mode

* Batch Training: All data is used to train the model in one go.

* Mini Batch Training: All the data is separated to multiple groups and the model will be trained based on the grouped data.

* Stochastic Training: The model will be trained on individual data.

# Getting Started

By default, the machine/deep learning models uses batch training upon initialization. To change this we create a TrainingModifier object.

We will modify the gradient descent to mini batch training.

```lua

local Model = DataPredict.Models.SupportVectorMachine.new()

local ModifiedModel = DataPredict.Others.TrainingModifier.new({Model = SupportVectorMachine, trainingMode = "MiniBatch"})

```

Once that is set up, you can call train() and predict() functions from the ModifiedModel. This is because it uses the original model's train() and predict() functions.

In other words, you can do what the original model can do, except the behaviour of the gradient descent and the cost has been changed.

```lua

local costArray = ModifiedModel:train(featureMatrix, labelVector)

local predictedVector = ModifiedModel:predict(featureMatrix2)

```

Looks pretty similar huh? You can try to combine this with other functionalities as well.

That's all for today!
