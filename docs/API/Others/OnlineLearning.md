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
* 
### stopOnlineLearning()

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
OnlineLearning:returnCostArrayFromOnlineLearningQueue(): number[]
```

#### Returns:

* costArray: The cost array related to actual value and the predicted value.
