# [API Reference](../../API.md) - [Models](../Models.md) - AsynchronousAdvantageCritic (A3C)

AsynchronousAdvantageCritic is a base class for reinforcement learning.

## Notes:

* The Actor and Critic child models must be created separately. Then use addActorCriticModel() to put it inside the AsynchronousAdvantageCritic model.

* Actor and Critic must be a part of NeuralNetwork model. If you decide to use linear regression or logistic regression, then it must be constructed using NeuralNetwork model. 

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
AsynchronousAdvantageCritic.new(numberOfReinforcementsPerEpisode: integer, epsilon: number, epsilonDecayFactor: number, discountFactor: number, rewardAveragingRate: number, totalNumberOfReinforcementsToUpdateMainModel: number): ModelObject
```

#### Parameters:

* numberOfReinforcementsPerEpisode: The number of reinforcements to decay the epsilon value. It will be also used for actor and critic loss calculations.

* epsilon: The higher the value, the more likely it focuses on exploration over exploitation. The value must be set between 0 and 1.

* epsilonDecayFactor: The higher the value, the slower the epsilon decays. The value must be set between 0 and 1.

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

* rewardAveragingRate: The higher the value, the higher the episodic reward, but lower the running reward.

* totalNumberOfReinforcementsToUpdateMainModel: The required total number of reinforce() function call from all child models to update the main model.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
AsynchronousAdvantageCritic:setParameters(numberOfReinforcementsPerEpisode: integer, epsilon: number, epsilonDecayFactor: number, discountFactor: number, rewardAveragingRate: number, totalNumberOfReinforcementsToUpdateMainModel: number)
```

#### Parameters:

* numberOfReinforcementsPerEpisode: The number of reinforcements to decay the epsilon value. It will be also used for actor and critic loss calculations.

* epsilon: The higher the value, the more likely it focuses on exploration over exploitation. The value must be set between 0 and 1.

* epsilonDecayFactor: The higher the value, the slower the epsilon decays. The value must be set between 0 and 1.

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

* rewardAveragingRate: The higher the value, the higher the episodic reward, but lower the running reward.

### addActorCriticModel()

```
AsynchronousAdvantageCritic:addActorCriticModel(ActorModel: ModelObject, CriticModel: ModelObject, ExperienceReplay: ExperienceReplayObject)
```

#### Parameters:

* ActorModel: The model to be used as an Actor model.

* CriticModel: The model to be used as a Critic model.

* ExperienceReplay: The experience replay object 

### setClassesList()

```
AsynchronousAdvantageCritic:setClassesList(classesList: [])
```

#### Parameters:

* classesList: A list of classes. The index of the class relates to which the neuron at output layer belong to. For example, {3, 1} means that the output for 3 is at first neuron, and the output for 1 is at second neuron.

### setActorCriticMainModelParameters()

```
AsynchronousAdvantageCritic:setActorCriticMainModelParameters(ActorMainModelParameters: [], CriticMainModelParameters[])
```

#### Parameters:

* ActorMainModelParameters: The model parameters to be set for main actor model.

* CriticMainModelParameters: The model parameters to be set for main critic model.

### getActorCriticMainModelParameters()

```
AsynchronousAdvantageCritic:getActorCriticMainModelParameters(): [], []
```

#### Returns:

* ActorMainModelParameters: The model parameters from the main actor model.

* CriticMainModelParameters: The model parameters from the main critic model.

### reinforce()

Reward or punish model based on the current state of the environment.

```
AsynchronousAdvantageCritic:reinforce(currentFeatureVector: Matrix, rewardValue: number, returnOriginalOutput: boolean, actorCriticModelNumber: number): integer, number -OR- Matrix
```

#### Parameters:

* currentFeatureVector: Matrix containing data from the current state.

* rewardValue: The reward value added/subtracted from the current state (recommended value between -1 and 1, but can be larger than these values). 

* returnOriginalOutput: Set whether or not to return predicted vector instead of value with highest probability.

* actorCriticModelNumber: The model number to be reinforced.

#### Returns:

* predictedLabel: A label that is predicted by the model.

* value: The value of predicted label.

-OR-

* predictedVector: A matrix containing all predicted values from all classes.

### update()

Updates the model parameters.

```
AsynchronousAdvantageCritic:update(previousFeatiureVector: featureVector, action: number/string, rewardValue: number, currentFeatureVector: featureVector, actorCriticModelNumber: number)
```

#### Parameters:

* previousFeatiureVector: The previous state of the environment.

* action: The action selected.

* rewardValue: The reward gained at current state.

* currentFeatureVector: The currrent state of the environment.

* actorCriticModelNumber: The model number to update the parameters.

### getCurrentNumberOfEpisodes()

```
AsynchronousAdvantageCritic:getCurrentNumberOfEpisodes(actorCriticModelNumber: number): number
```

#### Parameters:

* actorCriticModelNumber: The model number to get the current number of episodes.

#### Returns:

* currentNumberOfEpisodes: The current number of episodes.

### getCurrentNumberOfReinforcements()

```
AsynchronousAdvantageCritic:getCurrentNumberOfReinforcements(actorCriticModelNumber: number): number
```

#### Parameters:

* actorCriticModelNumber: The model number to get the current number of reinforcements.

#### Returns:

* numberOfReinforcements: The number of reinforce() funcion called.

### getCurrentEpsilon()

```
AsynchronousAdvantageCritic:getCurrentEpsilon(actorCriticModelNumber: number): number
```

#### Parameters:

* actorCriticModelNumber: The model number to get the epsilon.

#### Returns:

* currentEpsilon: The current epsilon.

### singleReset()

Reset reset a single child model's stored values (excluding the parameters).

```
AsynchronousAdvantageCritic:singleReset(actorCriticModelNumber)
```

#### Parameters:

* actorCriticModelNumber: The model number to apply the reset.

### reset()

Reset the main model's and child models' stored values (excluding the parameters).

```
AsynchronousAdvantageCritic:reset()
```

### destroy()

Destroys the model object.

```
AsynchronousAdvantageCritic:destroy()
```

### References

* [Advantage Actor Critic (A2C) implementation by Alvaro Durán Tovar](https://medium.com/deeplearningmadeeasy/advantage-actor-critic-a2c-implementation-944e98616b)

* [Actor Critic Method by Apoorv Nandan - Keras](https://keras.io/examples/rl/actor_critic_cartpole/)
