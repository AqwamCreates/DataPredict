# [API Reference](../../API.md) - [QuickSetups](../QuickSetups.md) - SingleCategoricalPolicy

SingleCategoricalPolicy is a base class for setuping up reinforcement learning functions.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
SingleCategoricalPolicy.new(numberOfReinforcementsPerEpisode: integer, epsilon: number, actionSelectionFunction: string, temperature: number): CategoricalPolicyObject
```

#### Parameters:

* numberOfReinforcementsPerEpisode: The number of reinforcements to be considered as a single episode.

* epsilon: The higher the value, the more likely it focuses on exploration over exploitation. The value must be set between 0 and 1. Exploration means choosing a random action to try to get better overall average performance, while exploiting means choosing an action that has the potential to increase the performance.

* actionSelectionFunction: The function on how to choose an action. Available options are:

  * Maximum (Default)

  * SoftmaxSampling / BoltzmannSampling

  * UpperConfidenceBound

* temperature: Controls the exploration and exploitation tradeoff. The higher the temperature, the more likely the model will do exploration. Only applicable when "SoftmaxSampling" or "BoltzmannSampling" is used.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
SingleCategoricalPolicy:setParameters(numberOfReinforcementsPerEpisode: integer, epsilon: number, actionSelectionFunction: string, temperature: number)
```

#### Parameters:

* numberOfReinforcementsPerEpisode: The number of reinforcements to decay the epsilon value.

* epsilon: The higher the value, the more likely it focuses on exploration over exploitation. The value must be set between 0 and 1.  Exploration means choosing a random action to try to get better overall average performance, while exploiting means choosing an action that has the potential to increase the performance.

* actionSelectionFunction: The function on how to choose an action. Available options are:

  * Maximum

  * SoftmaxSampling / BoltzmannSampling

  * UpperConfidenceBound

* temperature: Controls the exploration and exploitation tradeoff. The higher the temperature, the more likely the model will do exploration. Only applicable when "SoftmaxSampling" or "BoltzmannSampling" is used.

### setModel()

```
SingleCategoricalPolicy:setModel(Model: ModelObject)
```

#### Parameters:

* Model: The model object.

### getModel()

```
SingleCategoricalPolicy:getModel(): ModelObject
```

#### Returns:

* Model: The model object.

### setExperienceReplay()

```
SingleCategoricalPolicy:setExperienceReplay(ExperienceReplay: ExperienceReplayObject)
```

#### Parameters:

* ExperienceReplay: The experience replay object.

### getExperienceReplay()

```
SingleCategoricalPolicy:getExperienceReplay(): ExperienceReplayObject
```

#### Returns:

* ExperienceReplay: The experience replay object.

### setEpsilonValueScheduler()

```
SingleCategoricalPolicy:setEpsilonValueScheduler(EpsilonValueScheduler: ValueSchedulerObject)
```

#### Parameters:

* EpsilonValueScheduler: The value scheduler object to be used by the epsilon.

### getEpsilonValueScheduler()

```
SingleCategoricalPolicy:getEpsilonVaueScheduler(): ValueSchedulerObject
```

#### Returns:

* EpsilonValueScheduler: The value scheduler object that was used by the epsilon.

### setActionsList()

```
SingleCategoricalPolicy:setActionsList(classesList: {any})
```

#### Parameters:

* actionsList: A list of actions. The index of the action relates to which the neuron at output layer belong to. For example, {3, 1} means that the output for 3 is at first neuron, and the output for 1 is at second neuron.

### getClassesList()

Gets all the classes stored in the NeuralNetwork model.

```
SingleCategoricalPolicy:getActionsList(): {any}
```

#### Returns:

* actionsList: A list of actions. The index of the action relates to which the neuron at output layer belong to. For example, {3, 1} means that the output for 3 is at first neuron, and the output for 1 is at second neuron.

### extendUpdateFunction()

Sets a new function on update alongside with the current model's update() function. 

```
SingleCategoricalPolicy:extendUpdateFunction(updateFunction)
```

#### Parameters:

* updateFunction: The function to run after calling the model's update() function

### extendEpisodeUpdateFunction()

Sets a new function on episode update alongside with the current model's episodeUpdate() function. 

```
CategoricalPolicy:extendEpisodeUpdateFunction(episodeUpdateFunction)
```

#### Parameters:

* episodeUpdateFunction: The function to run after calling the model's episodeUpdate() function

### reinforce()

Reward or punish model based on the current state of the environment.

```
CategoricalPolicy:reinforce(currentFeatureVector: Matrix, rewardValue: number, returnOriginalOutput: boolean): integer, number -OR- Matrix
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

### reset()

Resets the current parameters values.

```
SingleCategoricalPolicy:reset()
```

### setPrintOutput()

Set whether or not to show the current number of episodes and current epsilon.

```
SingleCategoricalPolicy:setPrintOutput(option: boolean)
```

#### Parameters:

* option: A boolean value that determines the reinforcement output to be printed or not.

### getCurrentNumberOfEpisodes()

```
SingleCategoricalPolicy:getCurrentNumberOfEpisodes(): integer
```

#### Returns

* currentNumberOfEpisodes: The current number of episode stored inside the reinforcement learning quick setup object.

### getCurrentNumberOfReinforcements()

```
SingleCategoricalPolicy:getCurrentNumberOfReinforcements(): integer
```

#### Returns

* currentNumberOfReinforcements: The current number of times reinforce() has been called stored inside the reinforcement learning quick setup object.

### getCurrentEpsilon()

```
SingleCategoricalPolicy:getCurrentEpsilon(): number
```

#### Returns

* currentEpsilon: The current epsilon value stored inside the reinforcement learning quick setup object.


## Inherited From

* [CategoricalPolicyBaseQuickSetup](CategoricalPolicyBaseQuickSetup.md)
