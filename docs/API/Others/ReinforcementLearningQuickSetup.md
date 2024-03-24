# [API Reference](../../API.md) - [Others](../Others.md) - ReinforcementLearningQuickSetup

ReinforcementLearningQuickSetup is a base class for setuping up reinforcement learning functions.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
ReinforcementLearningQuickSetup.new(numberOfReinforcementsPerEpisode: integer, epsilon: number, epsilonDecayFactor: number, actionSelectionFunction: string): ReinforcementLearningQuickSetupObject
```

#### Parameters:

* numberOfReinforcementsPerEpisode: The number of reinforcements to decay the epsilon value.

* epsilon: The higher the value, the more likely it focuses on exploration over exploitation. The value must be set between 0 and 1.

* epsilonDecayFactor: The higher the value, the slower the epsilon decays. The value must be set between 0 and 1.

* actionSelectionFunction: The function on how to choose an action. Available options are:

  * Maximum (Default)

  * Sample 

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
ReinforcementLearningQuickSetup:setParameters(numberOfReinforcementsPerEpisode: integer, epsilon: number, epsilonDecayFactor: number, actionSelectionFunction: string
```

#### Parameters:

* numberOfReinforcementsPerEpisode: The number of reinforcements to decay the epsilon value.

* epsilon: The higher the value, the more likely it focuses on exploration over exploitation. The value must be set between 0 and 1.

* epsilonDecayFactor: The higher the value, the slower the epsilon decays. The value must be set between 0 and 1.

* actionSelectionFunction: The function on how to choose an action. Available options are:

  * Maximum

  * Sample 

### setModel()

```
ReinforcementLearningQuickSetup:setModel(Model: ModelObject)
```

#### Parameters:

* Model: The model object.

### getModel()

```
ReinforcementLearningQuickSetup:getModel(): ModelObject
```

#### Returns:

* Model: The model object.

### setExperienceReplay()

```
ReinforcementLearningQuickSetup:setExperienceReplay(ExperienceReplay: ExperienceReplayObject)
```

#### Parameters:

* ExperienceReplay: The experience replay object.

### getExperienceReplay()

```
ReinforcementLearningQuickSetup:getExperienceReplay(): ExperienceReplayObject
```

#### Returns:

* ExperienceReplay: The experience replay object.

### setClassesList()

```
NeuralNetwork:setClassesList(classesList: [])
```

#### Parameters:

* classesList: A list of classes. The index of the class relates to which the neuron at output layer belong to. For example, {3, 1} means that the output for 3 is at first neuron, and the output for 1 is at second neuron.

### getClassesList()

Gets all the classes stored in the NeuralNetwork model.

```
NeuralNetwork:getClassesList(): []
```

#### Returns:

* classesList: A list of classes. The index of the class relates to which the neuron at output layer belong to. For example, {3, 1} means that the output for 3 is at first neuron, and the output for 1 is at second neuron.

### reinforce()

Reward or punish model based on the current state of the environment.

```
ReinforcementLearningQuickSetup:reinforce(currentFeatureVector: Matrix, rewardValue: number, returnOriginalOutput: boolean): integer, number -OR- Matrix
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
ReinforcementLearningQuickSetup:reset()
```

### setPrintReinforcementOutput()

Set whether or not to show the current number of episodes and current epsilon.

```
ReinforcementLearningQuickSetup:setPrintReinforcementOutput(option: boolean)
```

#### Parameters:

* option: A boolean value that determines the reinforcement output to be printed or not.

### getCurrentNumberOfEpisodes()

```
ReinforcementLearningQuickSetup:getCurrentNumberOfEpisodes(): integer
```

#### Returns

* currentNumberOfEpisodes: The current number of episode stored inside the reinforcement learning quick setup object.

### getCurrentNumberOfReinforcements()

```
ReinforcementLearningQuickSetup:getCurrentNumberOfReinforcements(): integer
```

#### Returns

* currentNumberOfReinforcements: The current number of times reinforce() has been called stored inside the reinforcement learning quick setup object.

### getCurrentEpsilon()

```
ReinforcementLearningQuickSetup:getCurrentEpsilon(): number
```

#### Returns

* currentEpsilon: The current epsilon value stored inside the reinforcement learning quick setup object.

## Inherited From

* [NeuralNetwork](NeuralNetwork.md)
