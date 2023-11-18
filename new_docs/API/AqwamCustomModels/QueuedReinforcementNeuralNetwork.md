# [API Reference](../../API.md) - [AqwamCustomModels](../AqwamCustomModels.md) - QueuedReinforcementNeuralNetwork

A QueuedReinforcementNeuralNetwork (QR-NN) is a neural network that has the ability to learn through reinforcement and online learning. This model can be considered as a the sibling of Neural Network (NN).

It works by having a queue system for storing a number of variables.

First, we can add a feature vector to produce a prediction. The prediction is then added to queue as well as a number of other variables. 
Then once the actual value is given in the future, it will use the queued prediction with the actual prediction to "learn".

It has the ability to adapt real-time. It is also efficient amd flexible in many scenarios, especially for game development.

It was first discovered by Aqwam and does not exists in any research literature.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
QueuedReinforcementNeuralNetwork.new(maxNumberOfIterations: integer, learningRate: number, targetCost: number): ModelObject
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* targetCost: The cost at which the model stops training.

#### Returns:

* ModelObject: The generated model object.

## Functions

### start()

Creates new threads for real-time reinforcement.

```
QueuedReinforcementNeuralNetwork:start(rewardValue, punishValue, showPredictedLabel, showIdleWarning, showWaitingForLabelWarning): coroutine, coroutine, coroutine
```

#### Parameters:

* rewardValue: How much do we reward the model if it gets the prediction correct (value between 0 and 1).

* punishValue: How much do we punish the model if it gets the prediction incorrect (value between 0 and 1).

* showPredictedLabel: Set whether or not to show the predicted label and the actual label.

* showIdleWarning: Set whether or not to show idle warning if the thread has been idle for more than 30 seconds.

* showWaitingForLabelWarning: Set whether or not to show waiting for label warning if the thread has been waiting for more than 30 seconds.

#### Returns:

* predictCoroutine: A coroutine that produces prediction from the model.

* reinforcementCoroutine: A coroutine that controls the reinforcement behaviour.

* resetCoroutine: A coroutine where it resets the internal data. This is mainly to avoid memory leaks.

### stop()

Stops the threads for real-time reinforcement.

```
QueuedReinforcementNeuralNetwork:stop()
```

### addFeatureVector()

Adds feature vector to queue.

```
QueuedReinforcementNeuralNetwork:addFeatureVector(featureVector)
```

#### Parameters:

* featureVector: A (1 x n) matrix containing all the features to be added to the reinforcement queue.

### addLabel()

Adds label to queue.

```
QueuedReinforcementNeuralNetwork:addLabel(label)
```

#### Parameters:

* label: The actual label related to the previous feature vector.  

### returnPredictedLabel()

Returns predicted label from the queue.

```
QueuedReinforcementNeuralNetwork:returnPredictedLabel(): integer
```

#### Returns:

* label: The predicted label related to the previous feature vector.

### returnCost()

Returns predicted label from the queue.

```
QueuedReinforcementNeuralNetwork:returnCost(): number
```

#### Returns:

* cost: The cost related to actual value and the predicted value.

## Notes:

* It is recommended to train using some data before starting to use reinforcement on real-time data. This is because it will take too long for this model to learn by reinforcement. Training on some data allows the model to have some head-start.

* This model has the capability of reverting to previous model parameters if it starts to "unlearn". While it is a nice mechanism to avoid potential troubles on live games, I do recommend that you use appropriate reward and punish values.

## Inherited From

[NeuralNetwork](../Models/NeuralNetwork.md)
