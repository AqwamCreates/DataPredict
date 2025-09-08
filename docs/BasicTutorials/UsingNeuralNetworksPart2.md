# Using Neural Networks, Part 2

In part 1, you can see that we used the train() function to train our model. However, for some people, this can be inconvenient to use as it restricts the flexibility on designing our own models.
In this tutorial, we will show you another way of training our model.

We will use forwardPropagate() and backPropagate() functions to train our neural network model. We will show you a code sample with explanations below.

# Getting Started

```lua

local DataPredict = require(DataPredict)
local MatrixL = require(AqwamMatrixLibrary)

local featureMatrix = {

  {1, 2, 3, 4},
  {1, 2, 3, 4},

}

local labelMatrix = {
  
  {1, 0, 0, 0, 0, 0},
  {0, 0, 0, 0, 0, 1},

}

local NeuralNetwork = DataPredict.Model.NeuralNetwork.new() -- Creating our neural network model.

NeuralNetwork:addLayer(3, true, "None")

NeuralNetwork:addLayer(6, true, "LeakyReLU")

NeuralNetwork:addLayer(6, false, "LeakyReLU")

NeuralNetwork:setClassesList({1, 2, 3, 4, 5, 6})

-- As you can see the initial set up is the same as calling the train(). However, the difference can be seen at the code below.

local predictedMatrix = NeuralNetwork:forwardPropagate(featureMatrix, true)

local lossMatrix = MatrixL:subtract(predictedMatrix, labelMatrix)

local costFunctionDerivativesTable = NeuralNetwork:backwardPropagate(lossMatrix, true)

-- Unlike train(), we need to calculate the loss matrix before we can pass it to backwardPropagate() function.

```

In conclusion, the train() function will do the loss matrix calculations automatically. Meanwhile, this method requires us to calculate the loss function values.
