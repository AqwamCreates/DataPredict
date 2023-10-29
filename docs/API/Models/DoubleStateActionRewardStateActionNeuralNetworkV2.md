# [API Reference](../../API.md) - [Models](../Models.md) - DoubleStateActionRewardStateActionNV2 (Double Deep SARSA)

DoubleStateActionRewardStateActionV2 is a neural network with reinforcement learning capabilities. It can predict any positive numbers of discrete values. 

## Stored Model Parameters

Contains a table of matrices.  

* ModelParameters[L][I][J]: Matrix at layer L. Value of matrix at row I and column J.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
DoubleStateActionRewardStateActionV2.new(maxNumberOfIterations: integer, learningRate: number, targetCost: number, maxNumberOfEpisodes: integer, epsilon: number, epsilonDecayFactor: number, discountFactor: number, averagingRate: number): ModelObject
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* targetCost: The cost at which the model stops training.

* maxNumberOfEpisodes: The number of episodes to decay the epsilon value.

* epsilon: The higher the value, the more likely it focuses on exploration over exploitation. The value must be set between 0 and 1.

* epsilonDecayFactor: The higher the value, the slower the epsilon decays. The value must be set between 0 and 1.

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

* averagingRate: The lower the value, the faster the weights changes. The value must be set between 0 and 1.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
DoubleStateActionRewardStateActionV2:setParameters(maxNumberOfIterations: integer, learningRate: number, targetCost: number, maxNumberOfEpisodes: integer, epsilon: number, epsilonDecayFactor: number, discountFactor: number, averagingRate: number)
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* targetCost: The cost at which the model stops training.

* maxNumberOfEpisodes: The number of episodes to decay the epsilon value.

* epsilon: The higher the value, the more likely it focuses on exploration over exploitation. The value must be set between 0 and 1.

* epsilonDecayFactor: The higher the value, the slower the epsilon decays. The value must be set between 0 and 1.

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

* averagingRate: The lower the value, the faster the weights changes. The value must be set between 0 and 1.

## Inherited From

* [ReinforcementLearningNeuralNetworkBaseModel](ReinforcementLearningNeuralNetworkBaseModel.md)

## References

* [Double Deep Q Networks](https://towardsdatascience.com/double-deep-q-networks-905dd8325412)

* [Reinforcement Learning with Neural Network](https://www.baeldung.com/cs/reinforcement-learning-neural-network)

* [Deep Q Networks (DQN) in Python From Scratch by Using OpenAI Gym and TensorFlow- Reinforcement Learning Tutorial](https://aleksandarhaber.com/deep-q-networks-dqn-in-python-from-scratch-by-using-openai-gym-and-tensorflow-reinforcement-learning-tutorial/)

* [Reinforcement Q-Learning from Scratch in Python with OpenAI Gym](https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/)
