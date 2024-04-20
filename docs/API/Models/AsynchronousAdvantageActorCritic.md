# [API Reference](../../API.md) - [Models](../Models.md) - AsynchronousAdvantageActorCritic (A3C)

AsynchronousAdvantageCritic is a base class for reinforcement learning.

## Notes:

* The Actor and Critic child models must be created separately. Then use addActorCriticModel() to put it inside the AsynchronousAdvantageActorCritic model.

* Actor and Critic models must be a part of NeuralNetwork model. If you decide to use linear regression or logistic regression, then it must be constructed using NeuralNetwork model. 

* Ensure the final layer of the Critic model has only one neuron. It is the default setting for all Critic models in research papers.

* Ensure that setActorCriticMainModelParameters() is called first so that other child models can duplicate the main model parameters. Otherwise, the main model parameters will be selected randomly from the child models.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
AsynchronousAdvantageCritic.new(learningRate: integer, numberOfReinforcementsPerEpisode: integer, epsilon: number, epsilonDecayFactor: number, discountFactor: number, totalNumberOfReinforcementsToUpdateMainModel: number, actionSelectionFunction: string): ModelObject
```

#### Parameters:

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* numberOfReinforcementsPerEpisode: The number of reinforcements to decay the epsilon value. It will be also used for actor and critic loss calculations.

* epsilon: The higher the value, the more likely it focuses on exploration over exploitation. The value must be set between 0 and 1.

* epsilonDecayFactor: The higher the value, the slower the epsilon decays. The value must be set between 0 and 1.

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

* totalNumberOfReinforcementsToUpdateMainModel: The required total number of reinforce() function call from all child models to update the main model.

* actionSelectionFunction: The function on how to choose an action. Available options are:

    * Maximum (Default)

    * Sample (Requires the outputs to be in normal distribution format)

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
AsynchronousAdvantageCritic:setParameters(learningRate: integer, numberOfReinforcementsPerEpisode: integer, epsilon: number, epsilonDecayFactor: number, discountFactor: number, totalNumberOfReinforcementsToUpdateMainModel: number, actionSelectionFunction: string))
```

#### Parameters:

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* numberOfReinforcementsPerEpisode: The number of reinforcements to decay the epsilon value. It will be also used for actor and critic loss calculations.

* epsilon: The higher the value, the more likely it focuses on exploration over exploitation. The value must be set between 0 and 1.

* epsilonDecayFactor: The higher the value, the slower the epsilon decays. The value must be set between 0 and 1.

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

* totalNumberOfReinforcementsToUpdateMainModel: The required total number of reinforce() function call from all child models to update the main model.

* actionSelectionFunction: The function on how to choose an action. Available options are:

    * Maximum

    * Sample (Requires the outputs to be in normal distribution format)

### addActorCriticModel()

```
AsynchronousAdvantageCritic:addActorCriticModel(ActorModel: ModelObject, CriticModel: ModelObject, ExperienceReplay: ExperienceReplayObject)
```

#### Parameters:

* ActorModel: The model to be used as an Actor model.

* CriticModel: The model to be used as a Critic model.

* ExperienceReplay: The experience replay object.

### setClassesList()

```
AsynchronousAdvantageCritic:setClassesList(classesList: [])
```

#### Parameters:

* classesList: A list of classes. The index of the class relates to which the neuron at output layer belong to. For example, {3, 1} means that the output for 3 is at first neuron, and the output for 1 is at second neuron.

### setActorCriticMainModelParameters()

```
AsynchronousAdvantageCritic:setActorCriticMainModelParameters(ActorMainModelParameters: [], CriticMainModelParameters[], applyToAllChildModels: boolean)
```

#### Parameters:

* ActorMainModelParameters: The model parameters to be set for main actor model.

* CriticMainModelParameters: The model parameters to be set for main critic model.

* applyToAllChildModels: Set whether or not the main model parameters will be applied to all child models in the main model.

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

* actorCriticModelNumber: The model number for a model to be reinforced.

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

* actorCriticModelNumber: The model number for a model to update the parameters.

### getCurrentNumberOfEpisodes()

```
AsynchronousAdvantageCritic:getCurrentNumberOfEpisodes(actorCriticModelNumber: number): number
```

#### Parameters:

* actorCriticModelNumber: The model number for a model to get the current number of episodes.

#### Returns:

* currentNumberOfEpisodes: The current number of episodes.

### getCurrentNumberOfReinforcements()

```
AsynchronousAdvantageCritic:getCurrentNumberOfReinforcements(actorCriticModelNumber: number): number
```

#### Parameters:

* actorCriticModelNumber: The model number for a model to get the current number of reinforcements.

#### Returns:

* numberOfReinforcements: The number of reinforce() funcion called.

### getCurrentEpsilon()

```
AsynchronousAdvantageCritic:getCurrentEpsilon(actorCriticModelNumber: number): number
```

#### Parameters:

* actorCriticModelNumber: The model number for a model to get the epsilon.

#### Returns:

* currentEpsilon: The current epsilon.

### getCurrentTotalNumberOfReinforcementsToUpdateMainModel()

```
AsynchronousAdvantageCritic:getCurrentTotalNumberOfReinforcementsToUpdateMainModel(): number
```

#### Returns:

* getCurrentTotalNumberOfReinforcementsToUpdateMainModel: The current total number of reinforcements from all child models.


### singleReset()

Reset a single child model's stored values (excluding the parameters).

```
AsynchronousAdvantageCritic:singleReset(actorCriticModelNumber)
```

#### Parameters:

* actorCriticModelNumber: The model number for a model to be reset.

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

* [Asynchronous Methods for Deep Reinforcement Learning By Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, Koray Kavukcuoglu](https://arxiv.org/pdf/1602.01783v2.pdf)

* [Advantage Actor Critic (A2C) implementation by Alvaro Durán Tovar](https://medium.com/deeplearningmadeeasy/advantage-actor-critic-a2c-implementation-944e98616b)

* [Actor Critic Method by Apoorv Nandan - Keras](https://keras.io/examples/rl/actor_critic_cartpole/)
