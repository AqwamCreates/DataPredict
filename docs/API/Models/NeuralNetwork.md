# [API Reference](../../API.md) - [Models](../Models.md) - NeuralNetwork

NeuralNetwork is a supervised machine learning model that predicts any number of discrete values.

## Constructors

### new()

Create new model object. If any of the arguments are not given, default argument values for that argument will be used.

```
NeuralNetwork.new(maxNumberOfIterations: integer, learningRate: number, sigmoidFunction: string, targetCost: number): ModelObject
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* sigmoidFunction: The function to calculate the cost and cost derivaties of each training. Available options are "sigmoid", "ReLU", "LeakyReLU" and "ELU".

* targetCost: The cost at which the model stops training.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are not given, previous argument values for that argument will be used.

```
NeuralNetwork:setParameters(maxNumberOfIterations: integer, learningRate: number, sigmoidFunction: string, targetCost: number)
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* sigmoidFunction: The function to calculate the cost and cost derivaties of each training. Available options are "sigmoid", "ReLU", "LeakyReLU" and "ELU".

* targetCost: The cost at which the model stops training.

### setOptimizer()

Set optimizer for the model by inputting the optimizer object.

```
NeuralNetwork:setOptimizer(Optimizer: OptimizerObject)
```

#### Parameters:

* Optimizer: The optimizer to be used.

### setRegularization()

Set a regularization for the model by inputting the optimizer object.

```
NeuralNetwork:setRegularization(Regularization: RegularizationObject)
```

#### Parameters:

* Regularization: The regularization to be used.

### setLayers()

Set the number of layers and the neurons in each of thos layers. Number of arguments determines the layer, while the number value determines the number of neurons.

For example, setLayers(3,7,6) means 3 neurons at layer 1, 7 neurons at layer 2, and 6 neurons at layer 3.

```
NeuralNetwork:setLayers(...: integer)
```

#### Parameters:

* ...: layers and number of neurons.

### train()

Train the model.

```
NeuralNetwork:train(featureMatrix: Matrix, labelVector: Matrix): number[]
```
#### Parameters:

* featureMatrix: Matrix containing all data.

* labelVector: A (n x 1) matrix containing values related to featureMatrix.

#### Returns:

* costArray: An array containing cost values.

### predict()

* Predict the value for a given data.

```
NeuralNetwork:predict(featureMatrix: Matrix): integer
```

#### Parameters:

* featureMatrix: Matrix containing all data.

#### Returns:

* predictedValue: A value that is predicted by the model.

### reinforce()

Reward or punish model based on the predicted output.

```
NeuralNetwork:reinforce(featureVector: Matrix, label: integer, rewardValue: number, punishValue: number): integer
```

#### Parameters:

* featureVector: Matrix containing data.

* label: Actual label.

* rewardValue: How much do we reward the model if it gets the prediction correct (value between 0 and 1).

* punishValue: How much do we punish the model if it gets the prediction incorrect (value between 0 and 1).

#### Returns:

* predictedValue: A value that is predicted by the model.

## Inherited From

* [MachineLearningBaseModel](MachineLearningBaseModel.md)
