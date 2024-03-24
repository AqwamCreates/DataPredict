# [API Reference](../../API.md) - [Others](../Others.md) - ReinforcementLearningQuickSetup

ReinforcementLearningQuickSetup is a base class for reinforcement learning neural network models.

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

  * Maximum (Default)

  * Sample 

```
ReinforcementLearningQuickSetup:setExperienceReplay(ExperienceReplay: ExperienceReplayObject)
```

#### Parameters:

* ExperienceReplay: The experience replay object.

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

### setPrintReinforcementOutput()

Set whether or not to show the current number of episodes and current epsilon.

```
ReinforcementLearningNeuralNetworkBaseModel:setPrintReinforcementOutput(option: boolean)
```

#### Parameters:

* option: A boolean value that determines the reinforcement output to be printed or not.

## Inherited From

* [NeuralNetwork](NeuralNetwork.md)
