# [API Reference](../../API.md) - [Models](../Models.md) - NeuralNetwork

NeuralNetwork is a supervised machine learning model that predicts any positive numbers of discrete values.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
NeuralNetwork.new(maxNumberOfIterations: integer, learningRate: number, activationFunction: string, targetCost: number): ModelObject
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* activationFunction: The function to calculate the cost and cost derivaties of each training. Available options are "sigmoid", "ReLU", "LeakyReLU" and "ELU".

* targetCost: The cost at which the model stops training.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
NeuralNetwork:setParameters(maxNumberOfIterations: integer, learningRate: number, activationFunction: string, targetCost: number)
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* activationFunction: The function to calculate the cost and cost derivaties of each training. Available options are "sigmoid", "ReLU", "LeakyReLU" and "ELU".

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

Set the number of layers and the neurons (without bias neuron) in each of those layers. Number of arguments determines the layer, while the number value determines the number of neurons. Bias neurons will be added automatically after setting the number of neurons in each layer except for the output layer.

For example, setLayers(3,7,6) means 3 neurons at layer 1, 7 neurons at layer 2, and 6 neurons at layer 3. It assumes that the bias neurons are not counted. 

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

### startQueuedReinforcement()

Starts a new thread for real-time reinforcement. It waits for the functions below to provide both the feature vector and label, and only proceeds with reinforcement when both queues are filled.

```
NeuralNetwork:startQueuedReinforcement(rewardValue, punishValue, showPredictedLabel, showIdleWarning): coroutine
```

#### Parameters:

* rewardValue: How much do we reward the model if it gets the prediction correct (value between 0 and 1).

* punishValue: How much do we punish the model if it gets the prediction incorrect (value between 0 and 1).

* showPredictedLabel: Set whether or not to show the predicted label and the actual label.

* showIdleWarning: Set whether or not to show idle warning if he thread has been idle for more than 30 seconds.

#### Returns:

* queuedReinforcementCoroutine: A coroutine object.

### stopQueuedReinforcement()

Stops the thread for real-time reinforcement.

```
NeuralNetwork:stopQueuedReinforcement()
```

### addFeatureVectorToReinforcementQueue()

Adds feature vector to queue.

```
NeuralNetwork:addFeatureVectorToReinforcementQueue(featureVector)
```

#### Parameters:

* featureVector: A (1 x n) matrix containing all the features to be added to the reinforcement queue.

### addLabelToReinforcementQueue()

Adds label to queue.

```
NeuralNetwork:addLabelToReinforcementQueue(label)
```

#### Parameters:

* label: The actual label related to the previous feature vector.  

### getClassesList()

```
NeuralNetwork:getClassesList(): []
```

#### Returns:

* classesList: A list of classes. The index of the class relates to which the neuron at output layer belong to. For example, {3, 1} means that the output for 3 is at first neuron, and the output for 1 is at second neuron.

### setClassesList()

```
NeuralNetwork:setClassesList(classesList)
```

#### Parameters:

* classesList: A list of classes. The index of the class relates to which the neuron at output layer belong to. For example, {3, 1} means that the output for 3 is at first neuron, and the output for 1 is at second neuron. 

## Inherited From

* [BaseModel](BaseModel.md)
