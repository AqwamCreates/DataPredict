# [API Reference](../../API.md) - [Models](../Models.md) - NeuralNetwork

NeuralNetwork is a supervised machine learning model that predicts any positive numbers of discrete values.

## Stored Model Parameters

Contains a table of matrices.  

* ModelParameters[L][I][J]: Matrix at layer L. Value of matrix at row I and column J.

## Modded Models

* [QueuedReinforcementNeuralNetwork](../AqwamCustomModels/QueuedReinforcementNeuralNetwork.md)

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
NeuralNetwork.new(maxNumberOfIterations: integer, learningRate: number, targetCost: number): ModelObject
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* targetCost: The cost at which the model stops training.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
NeuralNetwork:setParameters(maxNumberOfIterations: integer, learningRate: number, targetCost: number)
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* targetCost: The cost at which the model stops training.

### addLayer()

Add a layer to the neural network.

```
NeuralNetwork:addLayer(numberOfNeurons: integer, hasBiasNeuron: boolean, activationFunction: string, Optimizer: OptimizerObject, Regularization: RegularizationObject)
```

#### Parameters:

* numberOfNeurons: Set the number of neurons to be added to the next layer (excluding bias neuron).

* hasBiasNeuron: Set whether or not the bias neuron will be added to next layer.

* activationFunction: The function to calculate the cost and cost derivaties of each training. Available options are:
  
  *  Sigmoid

  *  Tanh

  *  ReLU

  *  LeakyReLU

  *  ELU

  *  Softmax

  *  StableSoftmax

  *  None

* Optimizer: The optimizer object to be added at the last layer.

* Regularization: The regularization object to be added at the last layer.

### setLayer()

Change the properties of a selected layer of the neural netowrk.

```
NeuralNetwork:setLayer(layerNumber: integer, hasBiasNeuron: boolean, activationFunction: string, Optimizer: OptimizerObject, Regularization: RegularizationObject)
```

#### Parameters:

* layerNumber: The layer that you wish to change properties on.

* hasBiasNeuron: Set whether or not this layer has a bias neuron.

* activationFunction: The function to calculate the cost and cost derivaties of each training. Available options are:

  *  Sigmoid

  *  Tanh

  *  ReLU

  *  LeakyReLU

  *  ELU

  *  Softmax

  *  StableSoftmax

  *  None

* Optimizer: The optimizer object to be used by to this layer.

* Regularization: The regularization object to be used by to this layer.

### createLayers()

Create all the neurons (with bias neuron) in each of those layers. It also set all the activation function of all neuron to the activation function given in the function's parameters. Resets the current model parameters stored in the neural network.

```
NeuralNetwork:createLayers(numberOfNeuronsArray: integer[], activationFunction: string, Optimizer: OptimizerObject, Regularization: RegularizationObject)
```

#### Parameters:

* numberOfNeuronsArray: The array containing all the number of neurons for each layer (without bias neuron). The index determines the layer, while the value determines the number of neurons. Bias neurons will be added automatically after setting the number of neurons in each layer except for the output layer. For example, {3,7,6} means 3 neurons at layer 1, 7 neurons at layer 2, and 6 neurons at layer 3 wthout the bias neurons.

* activationFunction: The function to calculate the cost and cost derivaties of each training. Available options are "sigmoid", "tanh", "ReLU", "LeakyReLU" and "ELU".

* Optimizer: The optimizer object to be added at the last layer.

* Regularization: The regularization object to be added at the last layer.

### train()

Train the model.

```
NeuralNetwork:train(featureMatrix: Matrix, labelVector / labelMatrix: Matrix): number[]
```
#### Parameters:

* featureMatrix: Matrix containing all data.

* labelVector / labelMatrix: A (n x 1) / (n x o) matrix containing values related to featureMatrix. When using the label matrix, the number of columns must be equal to number of classes.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict the values for given data.

```
NeuralNetwork:predict(featureMatrix: Matrix, returnOriginalOutput: boolean): Matrix, Matrix -OR- Matrix
```

#### Parameters:

* featureMatrix: Matrix containing all data.

* returnOriginalOutput: Set whether or not to return predicted matrix instead of value with highest probability. 

#### Returns:

* predictedVector: A vector that is predicted by the model.

* probabilityVector: A vector that contains the probability of predicted values in predictedVector.

-OR-

* predictedMatrix: A matrix containing all predicted values from all classes.

### getClassesList()

```
NeuralNetwork:getClassesList(): []
```

#### Returns:

* classesList: A list of classes. The index of the class relates to which the neuron at output layer belong to. For example, {3, 1} means that the output for 3 is at first neuron, and the output for 1 is at second neuron.

### setClassesList()

```
NeuralNetwork:setClassesList(classesList: [])
```

#### Parameters:

* classesList: A list of classes. The index of the class relates to which the neuron at output layer belong to. For example, {3, 1} means that the output for 3 is at first neuron, and the output for 1 is at second neuron.

### showDetails()

Shows the details of all layers. The details includes the number of neurons, is bias added and so on.

```
NeuralNetwork:showDetails()
```

## Inherited From

* [BaseModel](BaseModel.md)
