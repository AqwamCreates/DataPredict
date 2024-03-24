# [API Reference](../../API.md) - [Models](../Models.md) - ReinforcementLearningNeuralNetworkBaseModel

ReinforcementLearningNeuralNetworkBaseModel is a base class for reinforcement learning neural network models.

## Stored Model Parameters

Contains a table of matrices.  

* ModelParameters[L][I][J]: Matrix at layer L. Value of matrix at row I and column J.

## Constructors

### new()

Creates a new base model object. If any of the arguments are nil, default argument values for that argument will be used.

```
ReinforcementLearningNeuralNetworkBaseModel.new(maxNumberOfIterations: integer, learningRate: number, discountFactor: number): ModelObject
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
ReinforcementLearningNeuralNetworkBaseModel:setParameters(maxNumberOfIterations: integer, learningRate: number, discountFactor: number)
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

### setUpdateFunction()

Sets the model's update function.

```
ReinforcementLearningNeuralNetworkBaseModel:setUpdateFunction(updateFunction)
```

#### Parameters:

* updateFunction: The function to run when update() is called.

### setEpisodeUpdateFunction()

Sets the model's episode update function.

```
ReinforcementLearningNeuralNetworkBaseModel:setEpisodeUpdateFunction(episodeUpdateFunction)
```

#### Parameters:

* episodeUpdateFunction: The function to run when episodeUpdate() is called.

### update()

Updates the model parameters using updateFunction().

```
ReinforcementLearningNeuralNetworkBaseModel:update(previousFeatiureVector: featureVector, action: number/string, rewardValue: number, currentFeatureVector: featureVector)
```

#### Parameters:

* previousFeatiureVector: The previous state of the environment.

* action: The action selected.

* rewardValue: The reward gained at current state.

* currentFeatureVector: The currrent state of the environment.

### episodeUpdate()

Updates the model parameters using episodeUpdateFunction().

```
ReinforcementLearningNeuralNetworkBaseModel:episodeUpdate()
```

### extendResetFunction()

Adds new function on reset alongside with the current reset() function. 

```
ReinforcementLearningNeuralNetworkBaseModel:extendResetFunction(resetFunction)
```

#### Parameters:

* resetFunction: The function to run when reset() is called.

### reset()

Reset model's stored values (excluding the parameters).

```
ReinforcementLearningNeuralNetworkBaseModel:reset()
```

## Inherited From

* [NeuralNetwork](NeuralNetwork.md)
