# [API Reference](../../API.md) - [Others](../Others.md) - OnlineLearning

Online machine learning continuously updates the model as it receives new data, making it capable of real-time training.

## Constructors

### new()

Creates a new online learning object

```
 OnlineLearning.new(Model: ModelObject, isLabelRequired: boolean, batchSize: integer): OnlineLearningObject
 
```

### Parameters:

* Model: The model to be trained.

* isLabelRequired: Set whether or not the model requires labels

* batchSize: The size of data needed before training the model.

## Functions

### startOnlineLearning()

Creates new threads for real-time training.

```
OnlineLearning:startOnlineLearning(showFinalCost: boolean, showWaitWarning: boolean): coroutine
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
OnlineLearning:addFeatureVectorToOnlineLearningQueue(featureVector: matrix)
```

#### Parameters:

* featureVector: A (1 x n) matrix containing all the features to be added to the reinforcement queue.

### addLabelToOnlineLearningQueue()

Adds label to queue.

```
OnlineLearning:addLabelToOnlineLearningQueue(label: integer)
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
