# What Are Neural Networks?

Neural networks are classification and regression model where it can predict a number of classes (classification) or range of values (regression).

Neural networks typically contains a number of layers. Inside each of those layers, they contain a number of neurons that determines the output of the model.

# Getting Started

Before we start building layers for our neural network, we first need to define a number of stuff in our neural network code.

```lua
local NeuralNetworkModel = DataPredict.Models.NeuralNetwork.new() -- Creating a new model object.

NeuralNetworkModel:setClassesList({0, 1}) -- Setting exising classes, but these can be automatically set by our model if the model uses batch gradient descent.
```

And under here, we have our data. Notice that all of our first column of the feature matrix contain values of 1. Those are bias values.

```lua
local featureMatrix = {

	{1, 0,  0},
	{1, 10, 2},
	{1, -3, -2},
	{1, -12, -22},
	{1,  2,  2},
	{1, 1,  1},
	{1, -11, -12},
	{1,  3,  3},
	{1, -2, -2},

}

local labelVector = {

	{1},
	{1},
	{0},
	{0},
	{1},
	{1},
	{0},
	{1},
	{0}

}
```

As you can see the feature matrix contains 3 columns, which means that we need 3 input neurons. It also contains the bias values, so one of the three input neurons must be a bias neuron.

# Creating Layers

We have two ways of creating our neural network layers:

1. Create all layers in one go.

2. Create each layer with their own individual settings.

Below, I will show the codes that demonstrates these two options

## Creating All Layers

We will use createLayers() function to create the layers. The first parameters takes in a table of integers, where the index determines the position and the values determines the number of neurons.

```lua
local numberOfNeuronsArray = {2, 3, 2}

NeuralNetworkModel:createLayers(numberOfNeuronsArray)
```

Using this function, we have set 2 neurons at first layer, 3 neurons at second layer and 2 neurons at final layer. 

Do make note that the bias neurons are not added yet to each of the layer (except the final layer) and will be added automatically once this function is called.

In other words, after running the function, the model will have 3 neurons at first layer, 4 neurons at second layer and 2 neurons at final layer.

## Creating Individual Layers

If you wish to have more control over each layer, then we can use addLayer() function. Below, we will show on how to create a single layer.

```lua
NeuralNetworkModel:addLayer(2, true, "Tanh")
```

The first parameter determines the number of neurons on that layer, the second parameter is to set whether or not to add a bias neuron. The third parameter is to set the activation function for that layer.

Do make note that if you add a bias neuron, it will not be included in the first parameter. (e.g. Before adding a bias neuron, it is 2 neuron, but after adding a bias neuron, it becomes three.)

Once that is covered, we will now show you on how to add multiple layers using the same function.

```lua
NeuralNetworkModel:addLayer(2, true, "Tanh")

NeuralNetworkModel:addLayer(3, true, "Tanh")

NeuralNetworkModel:addLayer(2, false, "StableSoftmax")
```

In this code, we have set 3 neurons (including the bias neuron) at first layer, 4 neurons (including the bias neuron) at second layer and 2 neurons (without the bias neuron) at final layer. 

# Wrapping it all up.

The tutorial covers the basics on how to create your neural networks. 

However, it does not cover the use of regularization and optimizers on each layer. These can be found in the API reference [here](../API/Models/NeuralNetwork.md).

That's all for today!
