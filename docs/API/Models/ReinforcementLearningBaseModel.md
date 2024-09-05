# [API Reference](../../API.md) - [Models](../Models.md) - ReinforcementLearningBaseModel

ReinforcementLearningBaseModel is a base class for reinforcement learning neural network models.

## Constructors

### new()

Creates a new base model object. If any of the arguments are nil, default argument values for that argument will be used.

```
ReinforcementLearningBaseModel.new(discountFactor: number): ModelObject
```

#### Parameters:

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
ReinforcementLearningBaseModel:setParameters(discountFactor: number)
```

### setModel()

Sets the model.

```
ReinforcementLearningBaseModel:setModel(Model: ModelObject)
```

#### Parameters:

* Model: The model to be used.

### getModel()

Gets the model.

```
ReinforcementLearningBaseModel:getModel(): ModelObject
```

#### Returns:

* Model: The model that was set.

### setCategoricalUpdateFunction()

Sets the model's categorical policy update function.

```
ReinforcementLearningBaseModel:setCategoricalUpdateFunction(categoricalUpdateFunction)
```

#### Parameters:

* categoricalUpdateFunction: The function to run when categoricalUpdate() is called.

### setDiagonalGaussianUpdateFunction()

Sets the model's diagonal Gausian policy update function.

```
ReinforcementLearningBaseModel:setDiagonalGaussianUpdateFunction(diagonalGaussianUpdateFunction)
```

#### Parameters:

* diagonalGaussianUpdateFunction: The function to run when diagonalGaussianUpdate() is called.

### setEpisodeUpdateFunction()

Sets the model's episode update function.

```
ReinforcementLearningBaseModel:setEpisodeUpdateFunction(episodeUpdateFunction)
```

#### Parameters:

* episodeUpdateFunction: The function to run when episodeUpdate() is called.

### categoricalUpdate()

Updates the model parameters using categoricalUpdateFunction().

```
ReinforcementLearningBaseModel:categoricalUpdate(previousFeatureVector: featureVector, action: number/string, rewardValue: number, currentFeatureVector: featureVector)
```

#### Parameters:

* previousFeatiureVector: The previous state of the environment.

* action: The action selected.

* rewardValue: The reward gained at current state.

* currentFeatureVector: The currrent state of the environment.

### diagonalGaussianUpdate()

Updates the model parameters using diagonalGaussianUpdateFunction().

```
ReinforcementLearningBaseModel:diagonalGaussianUpdate(previousFeatureVector: featureVector, actionVector: vector rewardValue: number, currentFeatureVector: featureVector)
```

#### Parameters:

* previousFeatiureVector: The previous state of the environment.

* actionVector: The action vector generated from the model.

* rewardValue: The reward gained at current state.

* currentFeatureVector: The currrent state of the environment.

### episodeUpdate()

Updates the model parameters using episodeUpdateFunction().

```
ReinforcementLearningBaseModel:episodeUpdate()
```

### setResetFunction()

Sets a new function on reset alongside with the current reset() function. 

```
ReinforcementLearningBaseModel:extendResetFunction(resetFunction)
```

#### Parameters:

* resetFunction: The function to run when reset() is called.

### reset()

Reset model's stored values (excluding the parameters).

```
ReinforcementLearningBaseModel:reset()
```
