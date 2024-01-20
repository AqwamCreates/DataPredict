# [API Reference](../../API.md) - [Models](../Models.md) - ReinforcementLearningActorCriticNeuralNetworkBaseModel

ReinforcementLearningActorCriticNeuralNetworkBaseModel is a base class for reinforcement learning neural network models.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
ReinforcementLearningActorCriticNeuralNetworkBaseModel.new(maxNumberOfEpisodes: integer, epsilon: number, epsilonDecayFactor: number, discountFactor: number): ModelObject
```

#### Parameters:

* maxNumberOfEpisodes: The number of episodes to decay the epsilon value.

* epsilon: The higher the value, the more likely it focuses on exploration over exploitation. The value must be set between 0 and 1.

* epsilonDecayFactor: The higher the value, the slower the epsilon decays. The value must be set between 0 and 1.

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
ReinforcementLearningActorCriticNeuralNetworkBaseModel:setParameters( maxNumberOfEpisodes: integer, epsilon: number, epsilonDecayFactor: number, discountFactor: number)
```

#### Parameters:

* maxNumberOfEpisodes: The number of episodes to decay the epsilon value.

* epsilon: The higher the value, the more likely it focuses on exploration over exploitation. The value must be set between 0 and 1.

* epsilonDecayFactor: The higher the value, the slower the epsilon decays. The value must be set between 0 and 1.

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

### setExperienceReplay()

Set model's settings for experience replay capabilities. When any parameters are set to nil, then it will use previous settings for that particular parameter.

```
ReinforcementLearningActorCriticNeuralNetworkBaseModel:setExperienceReplay(ExperienceReplay: ExperienceReplayObject)
```

#### Parameters:

* ExperienceReplay: The experience replay object.

### reinforce()

Reward or punish model based on the current state of the environment.

```
ReinforcementLearningActorCriticNeuralNetworkBaseModel:reinforce(currentFeatureVector: Matrix, rewardValue: number, returnOriginalOutput: boolean): integer, number -OR- Matrix
```

#### Parameters:

* currentFeatureVector: Matrix containing data from the current state.

* rewardValue: The reward value added/subtracted from the current state (recommended value between -1 and 1, but can be larger than these values). 

* returnOriginalOutput: Set whether or not to return predicted vector instead of value with highest probability.

#### Returns:

* predictedLabel: A label that is predicted by the model.

* value: The value of predicted label.

-OR-

* predictedVector: A matrix containing all predicted values from all classes.

### setPrintReinforcementOutput()

Set whether or not to show the current number of episodes and current epsilon.

```
ReinforcementLearningActorCriticNeuralNetworkBaseModel:setPrintReinforcementOutput(option: boolean)
```

#### Parameters:

* option: A boolean value that determines the reinforcement output to be printed or not.

### setUpdateFunction()

Sets the model's update function.

```
ReinforcementLearningActorCriticNeuralNetworkBaseModel:setUpdateFunction(updateFunction)
```

#### Parameters:

* updateFunction: The function to run when update() is called.

### setEpisodeUpdateFunction()

Sets the model's episode update function.

```
ReinforcementLearningActorCriticNeuralNetworkBaseModel:setEpisodeUpdateFunction(episodeUpdateFunction)
```

#### Parameters:

* episodeUpdateFunction: The function to run when episodeUpdate() is called.

### update()

Updates the model parameters using updateFunction().

```
ReinforcementLearningActorCriticNeuralNetworkBaseModel:update(previousFeatiureVector: featureVector, action: number/string, rewardValue: number, currentFeatureVector: featureVector)
```

#### Parameters:

* previousFeatiureVector: The previous state of the environment.

* action: The action selected.

* rewardValue: The reward gained at current state.

* currentFeatureVector: The currrent state of the environment.

### episodeUpdate()

Updates the model parameters using episodeUpdateFunction().

```
ReinforcementLearningActorCriticNeuralNetworkBaseModel:episodeUpdate()
```

### extendResetFunction()

Adds new function on reset alongside with the current reset() function. 

```
ReinforcementLearningActorCriticNeuralNetworkBaseModel:extendResetFunction(resetFunction)
```

#### Parameters:

* resetFunction: The function to run when reset() is called.

### reset()

Reset model's stored values (excluding the parameters).

```
ReinforcementLearningActorCriticNeuralNetworkBaseModel:reset()
```
