# [API Reference](../../API.md) - [Models](../Models.md) - AdvantageActorCritic (A2C)

AdvantageActorCritic is a base class for reinforcement learning.

## Notes:

* The Actor and Critic models must be created separately. Then use setActorModel() and setCriticModel() to put it inside the AdvantageActorCritic model.

* Actor and Critic models must be a part of NeuralNetwork model. If you decide to use linear regression or logistic regression, then it must be constructed using NeuralNetwork model. 

* Ensure the final layer of the Critic model has only one neuron. It is the default setting for all Critic models in research papers.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
AdvantageActorCritic.new(numberOfReinforcementsPerEpisode: integer, epsilon: number, epsilonDecayFactor: number, discountFactor: number, rewardAveragingRate: number): ModelObject
```

#### Parameters:

* numberOfReinforcementsPerEpisode: The number of reinforcements to decay the epsilon value. It will be also used for actor and critic loss calculations.

* epsilon: The higher the value, the more likely it focuses on exploration over exploitation. The value must be set between 0 and 1.

* epsilonDecayFactor: The higher the value, the slower the epsilon decays. The value must be set between 0 and 1.

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

* rewardAveragingRate: The higher the value, the higher the episodic reward, but lower the running reward.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
AdvantageActorCritic:setParameters(numberOfReinforcementsPerEpisode: integer, epsilon: number, epsilonDecayFactor: number, discountFactor: number, rewardAveragingRate: number)
```

#### Parameters:

* numberOfReinforcementsPerEpisode: The number of reinforcements to decay the epsilon value. It will be also used for actor and critic loss calculations.

* epsilon: The higher the value, the more likely it focuses on exploration over exploitation. The value must be set between 0 and 1.

* epsilonDecayFactor: The higher the value, the slower the epsilon decays. The value must be set between 0 and 1.

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

* rewardAveragingRate: The higher the value, the higher the episodic reward, but lower the running reward.

### setActorModel()

```
AdvantageActorCritic:setActorModel(Model: ModelObject)
```

#### Parameters:

* Model: The model to be used as an Actor model.

### setCriticModel()

```
AdvantageActorCritic:setActorModel(Model: ModelObject)
```

#### Parameters:

* Model: The model to be used as a Critic model.

### setExperienceReplay()

Set model's settings for experience replay capabilities. When any parameters are set to nil, then it will use previous settings for that particular parameter.

```
AdvantageActorCritic:setExperienceReplay(ExperienceReplay: ExperienceReplayObject)
```

#### Parameters:

* ExperienceReplay: The experience replay object.

### setClassesList()

```
AdvantageActorCritic:setClassesList(classesList: [])
```

#### Parameters:

* classesList: A list of classes. The index of the class relates to which the neuron at output layer belong to. For example, {3, 1} means that the output for 3 is at first neuron, and the output for 1 is at second neuron.

### reinforce()

Reward or punish model based on the current state of the environment.

```
AdvantageActorCritic:reinforce(currentFeatureVector: Matrix, rewardValue: number, returnOriginalOutput: boolean): integer, number -OR- Matrix
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
AdvantageActorCritic:setPrintReinforcementOutput(option: boolean)
```
#### Parameters:

* option: A boolean value that determines the reinforcement output to be printed or not.

### update()

Updates the model parameters.

```
AdvantageActorCritic:update(previousFeatiureVector: featureVector, action: number/string, rewardValue: number, currentFeatureVector: featureVector)
```

#### Parameters:

* previousFeatiureVector: The previous state of the environment.

* action: The action selected.

* rewardValue: The reward gained at current state.

* currentFeatureVector: The currrent state of the environment.

### getCurrentNumberOfEpisodes()

```
AdvantageActorCritic:getCurrentNumberOfEpisodes(): number
```

#### Returns:

* currentNumberOfEpisodes: The current number of episodes.

### getCurrentNumberOfReinforcements()

```
AdvantageActorCritic:getCurrentNumberOfReinforcements(): number
```

#### Returns:

* numberOfReinforcements: The number of reinforce() funcion called.

### getCurrentEpsilon()

```
AdvantageActorCritic:getCurrentEpsilon(): number
```

#### Returns:

* currentEpsilon: The current epsilon.

### reset()

Reset model's stored values (excluding the parameters).

```
AdvantageActorCritic:reset()
```

### destroy()

Destroys the model object.

```
AdvantageActorCritic:destroy()
```

### References

* [Advantage Actor Critic (A2C) implementation by Alvaro Durán Tovar](https://medium.com/deeplearningmadeeasy/advantage-actor-critic-a2c-implementation-944e98616b)

* [Actor Critic Method by Apoorv Nandan - Keras](https://keras.io/examples/rl/actor_critic_cartpole/)
