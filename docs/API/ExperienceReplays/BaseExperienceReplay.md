# [API Reference](../../API.md) - [ExperienceReplays](../ExperienceReplays.md) - BaseExperienceReplay

The base class for most experience replay classes. It serves as a template for using it with the reinforcement learning models.

## Constructors

### new()

Creates a new base experience replay object.

```
BaseExperienceReplay.new(batchSize: number, numberOfRunsToUpdate: number, maxBufferSize: number)
```

#### Parameters:

* batchSize: The number of experience to sample from for training.

* numberOfRunsToUpdate: The number of run() function needed to be called to run a single event of experience replay.

* maxBufferSize: The maximum number of experiences that can be kept inside the object.

## Functions

### setParameters()

Change the parameters of a base experience replay object.

```
BaseExperienceReplay:setParameters(batchSize: number, numberOfRunsToUpdate: number, maxBufferSize: number)
```

#### Parameters:

* batchSize: The number of experience to sample from for training.

* numberOfRunsToUpdate: The number of run() function needed to be called to run a single event of experience replay.

* maxBufferSize: The maximum number of experiences that can be kept inside the object.

### extendResetFunction()

Adds new function on reset alongside with the current reset() function.

```
BaseExperienceReplay:setResetFunction(resetFunction)
```

#### Parameters:

* resetFunction: The function to run when reset() is called.

### reset()

Resets the base experience replay object.

```
BaseExperienceReplay:reset()
```

### setRunFunction()

Sets the model's run function.

```
BaseExperienceReplay:setRunFunction(updateFunction)
```

#### Parameters:

* runFunction: The function to run when run() is called.

### run()

For every nth experience collected, it will run the updateFunction once. 

```
BaseExperienceReplay:run(updateFunction)
```

#### Parameters:

* updateFunction: The update function that updates a model.

### addExperience()

Adds an experience to the experiance replay object.

```
BaseExperienceReplay:addExperience(previousFeatureVector, action: number/string, rewardValue: number, currentFeatureVector)
```

#### Parameters:

* previousFeatureVector: The previous features of the environment.

* action: The action selected.

* rewardValue: The reward gained at current state.

* currentFeatureVector: The currrent features of the environment.

### addTemporalDifferenceError()

Adds a temporal difference error to the experiance replay object.

```
BaseExperienceReplay:addTemporalDifferenceError(temporalDifferenceErrorVectorOrValue)
```

#### Parameters:

* temporalDifferenceErrorVectorOrValue: The temporal difference error in a form of vector or value.

### setIsTemporalDifferenceErrorRequired()

Set whether or not to store temporal difference errors to the experiance replay object.

```
BaseExperienceReplay:setIsTemporalDifferenceErrorRequired(option)
```

#### Parameters:

* option: Set whether or not to store temporal difference errors.

## Inherited From

* [BaseInstance](../Cores/BaseInstance.md)