# [API Reference](../../API.md) - [Others](../Others.md) - OnlineLearning

Online learning allows models to update continuously as it receives new data, making it capable of real-time training.

## Constructors

### new()

Creates a new online learning object

```
 OnlineLearning.new(Model: ModelObject, isOutputRequired: boolean, batchSize: integer): OnlineLearningObject
```

### Parameters:

* Model: The model to be trained.

* isOutputRequired: Set whether or not the model requires labels / token output sequence arrays.

* batchSize: The size of data needed before training the model.

## Functions

### start()

Creates new threads for real-time training.

```
OnlineLearning:start(showFinalCost: boolean, showWaitWarning: boolean): coroutine
```

#### Parameters:

* showFinalCost: Set whether or not the final cost is displayed when training is complete.

* showWaitWarning: Set whether or not to show that the model have been waiting for data for more than 30 seconds.

#### Returns:

* trainCoroutine: A coroutine that trains the model.

### stop()

Stops the threads for real-time training.

```
OnlineLearning:stop()
```

### addInput()

Adds feature vector / token input sequence array to to queue.

```
OnlineLearning:addInput(input: matrix / tokenSequenceArray[])
```

#### Parameters:

* input: A (1 x n) matrix / a token input sequence array to be added to the reinforcement queue.

### addOutput()

Adds label / token output sequence array  to queue.

```
OnlineLearning:addOutput(output: integer / tokenSequenceArray[])
```

#### Parameters:

* output: The actual label related to the previous feature vector / a token output sequence array.  

### returnCostArray()

Returns cost array from the queue.

```
OnlineLearning:returnCostArray(): number[]
```

#### Returns:

* costArray: The cost array related to actual value and the predicted value.

## Inherited From

* [BaseInstance](../Cores/BaseInstance.md)

## Notes:

* Be aware that the model may suffer [concept drift](https://machinelearningmastery.com/gentle-introduction-concept-drift-machine-learning/) from training over long periods of time.
