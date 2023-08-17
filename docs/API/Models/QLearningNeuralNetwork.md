# [API Reference](../../API.md) - [Models](../Models.md) - QLearningNeuralNetwork

QLearningNeuralNetwork is a neural network with reinforcement learning capabilities. It can predict any positive numbers of discrete values.

## Stored Model Parameters

Contains a table of matrices.  

* ModelParameters[L][I][J]: Matrix at layer L. Value of matrix at row I and column J.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
QLearningNeuralNetwork.new(maxNumberOfIterations: integer, learningRate: number, targetCost: number, maxNumberOfEpisodes: integer, epsilon: number, epsilonDecayFactor: number, discountFactor: number): ModelObject
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* targetCost: The cost at which the model stops training.

* maxNumberOfEpisodes: Controls how well the model learns the best actions for maximizing rewards in an environment.

* epsilon: The higher the value, the more likely it focuses on exploration over exploitation. The value must be set between 0 and 1.

* epsilonDecayFactor: Controls how fast the model goes from exploring to exploiting as it learns. The value must be set between 0 and 1.

* discountFactor: Balances present and future rewards in agent decisions. 

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
QLearningNeuralNetwork:setParameters(maxNumberOfIterations: integer, learningRate: number, targetCost: number, maxNumberOfEpisodes: integer, epsilon: number, epsilonDecayFactor: number, discountFactor: number)
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* targetCost: The cost at which the model stops training.

* maxNumberOfEpisodes: Controls how well the model learns the best actions for maximizing rewards in an environment.

* epsilon: The higher the value, the more likely it focuses on exploration over exploitation. The value must be set between 0 and 1.

* epsilonDecayFactor: Controls how fast the model goes from exploring to exploiting as it learns. The value must be set between 0 and 1.

* discountFactor: Balances present and future rewards in agent decisions.

### setExperienceReplay()

Set model's settings for experience replay capabilities. When any parameters are set to nil, then it will use previous settings for that particular parameter.

```
QLearningNeuralNetwork:setExperienceReplay(useExperienceReplay: boolean, experienceReplayBatchSize: integer, numberOfReinforcementsForExperienceReplayUpdate: integer, maxExperienceReplayBufferSize: integer)
```

#### Parameters:

* useExperienceReplay: The option to set whether or not to enable experience reeplay. When enabled, it may require more resources.

* experienceReplayBatchSize: Determines how many of these experiences are sampled at once for batch training

* numberOfReinforcementsForExperienceReplayUpdate: How many times does the reinforce() function needed to be called in order to for a single update from experience replay.

* maxExperienceReplayBufferSize: The maximum size that the model can store the experiences.

### reinforce()

Reward or punish model based on the current state of the environment.

```
QLearningNeuralNetwork:reinforce(currentFeatureVector: Matrix, rewardValue: number, returnOriginalOutput: boolean): integer, number -OR- Matrix
```

#### Parameters:

* currentFeatureVector: Matrix containing data from the current state.

* rewardValue: The reward value added/subtracted from the current state (recommended value between -1 and 1, but can be larger than these values). 

* returnOriginalOutput: Set whether or not to return predicted vector instead of value with highest probability.

#### Returns:

* predictedValue: A value that is predicted by the model.

* probability: The probability of predicted value.

-OR-

* predictedVector: A matrix containing all predicted values from all classes.

### setPrintReinforcementOutput()

Set whether or not to show the current number of episodes and current epsilon.

```
QLearningNeuralNetwork:setPrintReinforcementOutput(option: boolean)
```

#### Parameters:

* option: A boolean value that determines the reinforcement output to be printed or not.

### reset()

Reset model's stored values (excluding the parameters).

```
QLearningNeuralNetwork:reset()
```

## Inherited From

* [NeuralNetwork](NeuralNetwork.md)

## References

* [Reinforcement Learning with Neural Network](https://www.baeldung.com/cs/reinforcement-learning-neural-network)

* [Deep Q Networks (DQN) in Python From Scratch by Using OpenAI Gym and TensorFlow- Reinforcement Learning Tutorial](https://aleksandarhaber.com/deep-q-networks-dqn-in-python-from-scratch-by-using-openai-gym-and-tensorflow-reinforcement-learning-tutorial/)

* [Reinforcement Q-Learning from Scratch in Python with OpenAI Gym](https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/)
