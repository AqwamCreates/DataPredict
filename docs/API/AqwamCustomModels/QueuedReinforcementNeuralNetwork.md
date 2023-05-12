# [API Reference](../../API.md) - [Models](../Models.md) - QueuedReinforcementNeuralNetwork

NeuralNetwork is a supervised machine learning model that predicts any positive numbers of discrete values.

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
