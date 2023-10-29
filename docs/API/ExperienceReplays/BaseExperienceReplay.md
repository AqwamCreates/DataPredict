# [API Reference](../../API.md) - [ExperienceReplays](../ExperienceReplays.md) - BaseExperienceReplay

The base class for most experience replay classes. It serves as a template for using it with the reinforcement learning models.

## Constructors

### new()

Creates a new base experience replay object.

```
BaseExperienceReplay.new(batchSize: number, numberOfExperienceToUpdate: number, maxBufferSize: number)
```

#### Parameters:

* batchSize: The number of experience to sample from for training.

* numberOfExperienceToUpdate: The number of experience needed for a single event of experience replay.

* maxBufferSize: The maximum number of experiences that can be kept inside the object.

## Functions

### setParameters()

Change the parameters of a base experience replay object.

```
BaseExperienceReplay:setParametersbatchSize: number, numberOfExperienceToUpdate: number, maxBufferSize: number)
```

#### Parameters:

* batchSize: The number of experience to sample from for training.

* numberOfExperienceToUpdate: The number of experience needed for a single event of experience replay.

* maxBufferSize: The maximum number of experiences that can be kept inside the object.

### setResetFunction()

Set the function to run when reset() function is called.

```
BaseExperienceReplay:setResetFunction(resetFunction)
```

#### Parameters:

* resetFunction: The function that resets the base experience replay object.

### reset()

Resets the base experience replay object.

```
BaseExperienceReplay:reset()
```

### setSampleFunction()

Set the function to run when sample() function is called.

```
BaseExperienceReplay:setSampleFunction(sampleFunction)
```

#### Parameters:

* setSampleFunction: The function that samples the experiences from the buffer in the base experience replay object.

### sample()

Samples a number of experiences based on batch size.

```
BaseExperienceReplay:sample()
```

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
BaseExperienceReplay:addExperience(previousState, action, rewardValue, currentState)
```

* previousState
