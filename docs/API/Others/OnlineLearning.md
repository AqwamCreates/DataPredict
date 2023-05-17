# [API Reference](../../API.md) - [AqwamCustomModels](../AqwamCustomModels.md) - QueuedReinforcementNeuralNetwork

A QueuedReinforcementNeuralNetwork (QR-NN) is a neural network that has the ability to learn through reinforcement and online learning. This model can be considered as a the sibling of Neural Network (NN).

It works by having a queue system for storing a number of variables.

First, we can add a feature vector to produce a prediction. The prediction is then added to queue as well as a number of other variables. 
Then once the actual value is given in the future, it will use the queued prediction with the actual prediction to "learn".

It has the ability to adapt real-time. It is also efficient amd flexible in many scenarios, especially for game development.

It was first discovered by Aqwam and does not exists in any research literature.

## Functions

### startOnlineLearning()

Creates new threads for real-time training.

```
OnlineLearning:startOnlineLearning(showFinalCost, showWaitWarning): coroutine
```

#### Parameters:

* showFinalCost: Set whether or not the final cost 

* showWaitWarning: How much do we punish the model if it gets the prediction incorrect (value between 0 and 1).

#### Returns:

* trainCoroutine: A coroutine that trains the model.
### stopQueuedReinforcement()

Stops the threads for real-time training.

```
OnlineLearning:stopOnlineLearning()
```

### addFeatureVectorToOnlineLearningQueue()

Adds feature vector to queue.

```
OnlineLearning:addFeatureVectorToOnlineLearningQueue(featureVector)
```

#### Parameters:

* featureVector: A (1 x n) matrix containing all the features to be added to the reinforcement queue.

### addLabelToOnlineLearningQueue()

Adds label to queue.

```
OnlineLearning:addLabelToOnlineLearningQueue(label)
```

#### Parameters:

* label: The actual label related to the previous feature vector.  

### returnCostArrayFromOnlineLearningQueue()

Returns cost array from the queue.

```
OnlineLearning:returnCostArrayFromOnlineLearningQueue(): number
```

#### Returns:

* costArray: The cost array related to actual value and the predicted value.

## Notes:

* It is recommended to train using some data before starting to use reinforcement on real-time data. This is because it will take too long for this model to learn by reinforcement. Training on some data allows the model to have some head-start.

* This model has the capability of reverting to previous model parameters if it starts to "unlearn". While it is a nice mechanism to avoid potential troubles on live games, I do recommend that you use appropriate reward and punish values.

## Inherited From:

[NeuralNetwork](../Models/NeuralNetwork.md)
