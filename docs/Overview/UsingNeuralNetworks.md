# What Are Neural Networks?

Neural networks are classification and regression model where it can predict a number of classes (classification) or range of values (regression).

Neural networks typically contains a number of layers. Inside those each layer, the ycontain a number of neurons that determines the output of the model.

# Getting Started

Before we start building layers for our neural network, we first need to define a number of stuff in our neural network code.

```
local NeuralNetworkModel = DataPredict.Models.NeuralNetwork.new() -- Creating a new model object.

NeuralNetworkModel:setClassesList({0, 1}) -- Setting exising classes, but these can be automatically set by our model if the model uses batch gradient descent.
```
