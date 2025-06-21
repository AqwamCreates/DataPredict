# [API Reference](../../API.md) - [Models](../Models.md) - NeuralNetwork

NeuralNetwork is a supervised machine learning model that predicts any positive numbers of discrete or continuous values.

## Stored Model Parameters

Contains a table of matrices.  

* ModelParameters[L][I][J]: Matrix at layer L. Value of matrix at row I and column J. The row I indicates the number of neurons in the previous layer, while column J indicates the number of neurons in the next layer.

## Notes

* Only numberOfNeurons and hasBiasNeurons parameters are used for the first layer.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
NeuralNetwork.new(maximumNumberOfIterations: integer, costFunction: string): ModelObject
```

#### Parameters:

* maximumNumberOfIterations: How many times should the model needed to be trained. [Default: 500]

* costFunction: The function to calculate the cost of each training. Available options are:

	* MeanSquaredError (Default)

	* MeanAbsoluteError

	* BinaryCrossEntropy

	* CategoricalCrossEntropy

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
NeuralNetwork:setParameters(maximumNumberOfIterations: integer, costFunction: string)
```

#### Parameters:

* maximumNumberOfIterations: How many times should the model needed to be trained.

* costFunction: The function to calculate the cost of each training. Available options are:

	* MeanSquaredError (Default)

	* MeanAbsoluteError

	* BinaryCrossEntropy

	* CategoricalCrossEntropy

### addLayer()

Add a layer to the neural network.

```
NeuralNetwork:addLayer(numberOfNeurons: integer, hasBiasNeuron: boolean, activationFunction: string, learningRate: number, Optimizer: OptimizerObject, Regularization: RegularizationObject. dropoutRate: number)
```

#### Parameters:

* numberOfNeurons: Set the number of neurons to be added to the next layer (excluding bias neuron).

* hasBiasNeuron: Set whether or not the bias neuron will be added to next layer.

* activationFunction: The activation function to be used for the next layer. Available options are:
  
  * Sigmoid

  * Tanh

  * ReLU

  * LeakyReLU

  * ELU

  * Gaussian
 
  * SiLU

  * Mish

  * BinaryStep

  * Softmax

  * StableSoftmax

  * None

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* Optimizer: The optimizer object to be added at the last layer.

* Regularization: The regularization object to be added at the last layer.

* dropoutRate: The probabiliy of a neuron for selected layer number to be dropped out when required. Must be set between 0 and 1. Increasing the rate will cause more neurons more likely to output 0. By default, the dropoutRate is set to 0.

### setLayer()

Change the properties of a selected layer of the neural netowrk.

```
NeuralNetwork:setLayer(layerNumber: integer, hasBiasNeuron: boolean, activationFunction: string, learningRate: number, Optimizer: OptimizerObject, Regularization: RegularizationObject, dropoutRate: number)
```

#### Parameters:

* layerNumber: The layer that you wish to change properties on.

* hasBiasNeuron: Set whether or not this layer has a bias neuron.

* activationFunction: The activation function to be used for the current layer. Available options are:

  * Sigmoid

  * Tanh

  * ReLU

  * LeakyReLU

  * ELU

  * Gaussian
 
  * SiLU

  * Mish

  * BinaryStep

  * Softmax

  * StableSoftmax

  * None

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* Optimizer: The optimizer object to be used by to this layer.

* Regularization: The regularization object to be used by to this layer.

* dropoutRate: The probabiliy of a neuron for selected layer number to be dropped out when required. Must be set between 0 and 1. Increasing the rate will cause more neurons more likely to output 0. By default, the dropoutRate is set to 0.

### createLayers()

Create all the neurons (with bias neuron) in each of those layers. It also set all the activation function of all neuron to the activation function given in the function's parameters. Resets the current model parameters stored in the neural network.

```
NeuralNetwork:createLayers(numberOfNeuronsArray: integer[], activationFunction: string, learningRate, Optimizer: OptimizerObject, Regularization: RegularizationObject, dropoutRate: number)
```

#### Parameters:

* numberOfNeuronsArray: The array containing all the number of neurons for each layer (without bias neuron). The index determines the layer, while the value determines the number of neurons. Bias neurons will be added automatically after setting the number of neurons in each layer except for the output layer. For example, {3,7,6} means 3 neurons at layer 1, 7 neurons at layer 2, and 6 neurons at layer 3 wthout the bias neurons.

* activationFunction: The activation function to be used for all layers. Available options are:

  * Sigmoid

  * Tanh

  * ReLU

  * LeakyReLU

  * ELU

  * Gaussian
 
  * SiLU

  * Mish

  * BinaryStep

  * Softmax

  * StableSoftmax

  * None

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* Optimizer: The optimizer object to be added at the last layer.

* Regularization: The regularization object to be added at the last layer.

* dropoutRate: The probabiliy of a neuron for selected layer number to be dropped out when required. Must be set between 0 and 1. Increasing the rate will cause more neurons more likely to output 0. By default, the dropoutRate is set to 0.

### setLayerProperty()

```
NeuralNetwork:setLayerProperty(layerNumber: integer, property: string, value: any)
```

### Parameters:

* layerNumber: The layer to change its properties.

* property: The property to be changed. The options available are:

  * HasBias

  * ActivationFunction

  * LearningRate

  * DropoutRate

  * Optimizer
    
  * Regularization
    
  * DropoutRate

* value: The value to be set to the property for selected layer.

### getLayerProperty()

```
NeuralNetwork:getLayerProperty(layerNumber: integer, property: string): any
```

### Parameters:

* layerNumber: The layer to retrieve its properties.

* property: The property to be changed. The options available are:

  * HasBias

  * ActivationFunction

  * LearningRate
    
  * Optimizer
    
  * Regularization
    
  * DropoutRate

### Returns:

* value: The value for that particular property for selected layer.

### evolveLayerSize()

Evolves a specified layer by changing the number of neurons.

```
NeuralNetwork:evolveLayerSize(layerNumber: number, initialNeuronIndex: number, size: number)
```

#### Parameters:

* layerNumber: The layer to evolve.

* initialNeuronIndex: The starting point where to add or remove new neurons. When removing neurons, this will be the first neuron to be removed. When adding neurons, the new neurons will be added after this neuron. Setting this to nil will cause it to choose the final neuron.

* size: The number of neurons to add or to remove. Positive value indicates the addition of new neurons, while negative values indicates the removal of current neurons.

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

* predictedlabelVector: A vector tcontaining predicted labels generated from the model.

* valueVector: A vector that contains the values of predicted labels.

-OR-

* predictedMatrix: A matrix containing all predicted values from all classes.

### getClassesList()

Gets all the classes stored in the NeuralNetwork model.

```
NeuralNetwork:getClassesList(): []
```

#### Returns:

* ClassesList: A list of classes. The index of the class relates to which the neuron at output layer belong to. For example, {3, 1} means that the output for 3 is at first neuron, and the output for 1 is at second neuron.

### setClassesList()

```
NeuralNetwork:setClassesList(ClassesList: [])
```

#### Parameters:

* ClassesList: A list of classes. The index of the class relates to which the neuron at output layer belong to. For example, {3, 1} means that the output for 3 is at first neuron, and the output for 1 is at second neuron.

### forwardPropagate()

```
NeuralNetwork:forwardPropagate(featureMatrix: Matrix, saveAllArrays: boolean, doNotDropoutNeurons: boolean): predictedLabelMatrix, forwardPropagateTable, zTable
```

### Parameters:

* featureMatrix: Matrix containing all data.

* saveAllArrays: Set whether or not the forward propagation array and z array is stored in the model.

* doNotDropoutNeurons: Set whether or not to dropout neurons during forward propagation.

### Returns:

* predictedLabelMatrix: A matrix containing final layer outputs.

* forwardPropagateTable: A table containing matrices where its original values are transformed by selected activation functions.

* zTable: A table containing matrices that was produced by each neuron.

### update()

```
NeuralNetwork:update(lossMatrix: Matrix, clearAllArrays: boolean): []
```

### Parameters:

* lossMatrix: Matrix containing the loss.

* clearAllArrays: Set whether or not to clear forward propagation array and z array is stored in the model after backpropagation is done.

### Returns:

* costFunctionDerivativesTable: A table of matrices containing the model's weights cost function derivatives values.

### showDetails()

Shows the details of all layers. The details includes the number of neurons, is bias added and so on.

```
NeuralNetwork:showDetails()
```

### getLayer()

Gets the settings of a particular layer.

```
NeuralNetwork:getLayer(layerNumber: number): number, boolean, string, number, OptimizerObject, RegularizationObject, number
```

#### Parameters:

* layerNumber: The layer number to retrieve its properties.

#### Returns:

* numberOfNeurons: The number of neurons at that particular layer.

* activationFunction: The function to calculate the cost and cost derivaties of each training.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* Optimizer: The optimizer object to be added at the last layer.

* Regularization: The regularization object to be added at the last layer.

* dropoutRate: The probabiliy of a neuron for selected layer number to be dropped out when required. Must be set between 0 and 1. Increasing the rate will cause more neurons more likely to output 0. By default, the dropoutRate is set to 0.

### getNumberOfLayers()

Gets the number of layers.

```
NeuralNetwork:getNumberOfLayers(): number
```

#### Returns:

* numberOfLayers: The number of layers contained in the NeuralNetwork model.

### getTotalNumberOfNeurons()

Gets the total number of neurons (including the bias if present) at the selected layer number.

```
NeuralNetwork:getTotalNumberOfNeurons(layerNumber: number): number
```

#### Returns:

* layerNumber: The layer number to fetch the total number of neurons.

#### Returns:

* totalNumberOfNeurons: The number of neurons including the bias.

## Inherited From

* [GradientMethodBaseModel](GradientMethodBaseModel.md)

## References

* [The Derivative Of Softmax Z Function By Mehran](https://www.mldawn.com/the-derivative-of-softmaxz-function-w-r-t-z/)

* [Dropout in Neural Networks By Harsh Yadav](https://towardsdatascience.com/dropout-in-neural-networks-47a162d621d9)
