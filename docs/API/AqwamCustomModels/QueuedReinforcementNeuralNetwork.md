# [API Reference](../../API.md) - [Models](../Models.md) - [NeuralNetwork](../Models/NeuralNetwork.md) - QueuedReinforcementNeuralNetwork

A Queued Reinforcement Neural Network (QR-NN) is a neural network that has the capability of reinforcing and back propagate through time. This model can be considered as a cousin of Recurrent Neural Networks.

It works by having a queue system for storing a number of variables.

First, we can add a feature vector to produce a prediction. The prediction is then added to queue as well as a number of other variables. 
Then once the actual value is given in the future, it will use the queued prediction with the actual prediction to "learn".

It has the ability to adapt real-time.

It was first discovered by Aqwam and does not exists in any research literature.

## Functions

### startQueuedReinforcement()

```
NeuralNetwork:startQueuedReinforcement(rewardValue, punishValue, showPredictedLabel, showIdleWarning, showWaitingForLabelWarning): coroutine, coroutine, coroutine
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

### returnPredictedLabelFromReinforcementQueue()

Returns predicted label from the queue.

```
NeuralNetwork:returnPredictedLabelFromReinforcementQueue(): integer
```

#### Returns:

* label: The predicted label related to the previous feature vector.  
