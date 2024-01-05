# Saving And Loading Model Parameters

DataPredict provides the ability to save and load model parameters from trained models. The only requirement is that the model must inherit from the BaseModel class. You can find which models are inherited by BaseModel in the API Reference.

# Saving

In order to save the model parameters, we first need to call the getModelParameters() function on our model.

```lua

local SavedModelParameters = NeuralNetwork:getModelParameters()

```

This should make a deep copy of the model parameters to SavedModelParameters variable.

Now, do make note that different models stores different model parameter structures. You can have a look at the model parameters structures at the top page of each model. In this case, the NeuralNetwork stores a table of matrices.

You have two ways of saving the model parameters:

1. Storing it to DataStores.

2. Print out the matrices using printPortableMatrix() from MatrixL / Aqwam's Matrix Library and copy paste the text to a new text file. Don't forget to remove the unncessary lines produced by the console.

# Loading

To load a model parameters to the model, all you need to do is to call the setModelParameters() function on our model.

```lua

NeuralNetwork:setModelParameters(SavedModelParameters)

```

Make sure the model parameters structure are the same as shown as in the API reference. Otherwise, it will break the model when you try to run it.

# Wrapping up

Saving and loading on DataPredict has never been easier. All you need is to call few lines of codes and you're off!.

That's all you need to do. Pretty simple, right?

Thank you very much for reading this tutorial.
