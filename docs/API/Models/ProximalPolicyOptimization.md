# [API Reference](../../API.md) - [Models](../Models.md) - ProximalPolicyOptimization (PPO)

ProximalPolicyOptimization is a base class for reinforcement learning.

## Notes:

* The Actor and Critic models must be created separately. Then use setActorModel() and setCriticModel() to put it inside the AdvantageActorCritic model.

* Actor and Critic models must be a part of NeuralNetwork model. If you decide to use linear regression or logistic regression, then it must be constructed using NeuralNetwork model. 

* Ensure the final layer of the Critic model has only one neuron. It is the default setting for all Critic models in research papers.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
ProximalPolicyOptimization.new(numberOfReinforcementsPerEpisode: integer, epsilon: number, epsilonDecayFactor: number, discountFactor: number): ModelObject
```

#### Parameters:

* numberOfReinforcementsPerEpisode: The number of reinforcements to decay the epsilon value. It will be also used for actor and critic loss calculations.

* epsilon: The higher the value, the more likely it focuses on exploration over exploitation. The value must be set between 0 and 1.

* epsilonDecayFactor: The higher the value, the slower the epsilon decays. The value must be set between 0 and 1.

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
ProximalPolicyOptimization:setParameters(numberOfReinforcementsPerEpisode: integer, epsilon: number, epsilonDecayFactor: number, discountFactor: number)
```

#### Parameters:

* numberOfReinforcementsPerEpisode: The number of reinforcements to decay the epsilon value. It will be also used for actor and critic loss calculations.

* epsilon: The higher the value, the more likely it focuses on exploration over exploitation. The value must be set between 0 and 1.

* epsilonDecayFactor: The higher the value, the slower the epsilon decays. The value must be set between 0 and 1.

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

* rewardAveragingRate: The higher the value, the higher the episodic reward, but lower the running reward.

### setActorModel()

```
ProximalPolicyOptimization:setActorModel(Model: ModelObject)
```

#### Parameters:

* Model: The model to be used as an Actor model.

### setCriticModel()

```
ProximalPolicyOptimization:setActorModel(Model: ModelObject)
```

#### Parameters:

* Model: The model to be used as a Critic model.

### setExperienceReplay()

Set model's settings for experience replay capabilities. When any parameters are set to nil, then it will use previous settings for that particular parameter.

```
ProximalPolicyOptimization:setExperienceReplay(ExperienceReplay: ExperienceReplayObject)
```

#### Parameters:

* ExperienceReplay: The experience replay object.

### setClassesList()

```
ProximalPolicyOptimization:setClassesList(classesList: [])
```

#### Parameters:

* classesList: A list of classes. The index of the class relates to which the neuron at output layer belong to. For example, {3, 1} means that the output for 3 is at first neuron, and the output for 1 is at second neuron.

### reinforce()

Reward or punish model based on the current state of the environment.

```
ProximalPolicyOptimization:reinforce(currentFeatureVector: Matrix, rewardValue: number, returnOriginalOutput: boolean): integer, number -OR- Matrix
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
ProximalPolicyOptimization:setPrintReinforcementOutput(option: boolean)
```
#### Parameters:

* option: A boolean value that determines the reinforcement output to be printed or not.

### update()

Updates the model parameters.

```
ProximalPolicyOptimization:update(previousFeatiureVector: featureVector, action: number/string, rewardValue: number, currentFeatureVector: featureVector)
```

#### Parameters:

* previousFeatiureVector: The previous state of the environment.

* action: The action selected.

* rewardValue: The reward gained at current state.

* currentFeatureVector: The currrent state of the environment.

### getCurrentNumberOfEpisodes()

```
ProximalPolicyOptimization:getCurrentNumberOfEpisodes(): number
```

#### Returns:

* currentNumberOfEpisodes: The current number of episodes.

### getCurrentNumberOfReinforcements()

```
ProximalPolicyOptimization:getCurrentNumberOfReinforcements(): number
```

#### Returns:

* numberOfReinforcements: The number of reinforce() funcion called.

### getCurrentEpsilon()

```
ProximalPolicyOptimization:getCurrentEpsilon(): number
```

#### Returns:

* currentEpsilon: The current epsilon.

### reset()

Reset model's stored values (excluding the parameters).

```
ProximalPolicyOptimization:reset()
```

### destroy()

Destroys the model object.

```
ProximalPolicyOptimization:destroy()
```

### References

* [Proximal Policy Optimization By OpenAI](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

* [PPO Python Code By nikhilbarhate99](https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py)
