# [API Reference](../../API.md) - [Models](../Models.md) - StateActionRewardStateActionNeuralNetwork

StateActionRewardStateActionNeuralNetwork is a neural network with reinforcing learning capabilities. It can predict any positive numbers of discrete values.

## Stored Model Parameters

Contains a table of matrices.  

* ModelParameters[L][I][J]: Matrix at layer L. Value of matrix at row I and column J.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
StateActionRewardStateActionNeuralNetwork.new(maxNumberOfIterations: integer, learningRate: number, targetCost: number, maxNumberOfEpisodes: integer, epsilon: number, epsilonDecayFactor: number, discountFactor: number): ModelObject
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* targetCost: The cost at which the model stops training.

* maxNumberOfEpisodes: Controls how well the model learns the best actions for maximizing rewards in an environment.

* epsilon: The higher the value, the more likely it focuses on exploration over exploitation.

* epsilonDecayFactor: Controls how fast the model goes from exploring to exploiting as it learns.

* discountFactor: Balances present and future rewards in agent decisions.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
StateActionRewardStateActionNeuralNetwork:setParameters(maxNumberOfIterations: integer, learningRate: number, targetCost: number, maxNumberOfEpisodes: integer, epsilon: number, epsilonDecayFactor: number, discountFactor: number)
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* targetCost: The cost at which the model stops training.

* maxNumberOfEpisodes: Controls how well the model learns the best actions for maximizing rewards in an environment.

* epsilon: The higher the value, the more likely it focuses on exploration over exploitation.

* epsilonDecayFactor: Controls how fast the model goes from exploring to exploiting as it learns.

* discountFactor: Balances present and future rewards in agent decisions.

### reinforce()

Reward or punish model based on the predicted output.

```
StateActionRewardStateActionNeuralNetwork:reinforce(currentFeatureVector: Matrix, rewardValue: number): integer, number
```

#### Parameters:

* currentFeatureVector: Matrix containing data for current state.

* rewardValue: The reward value added/subtracted during current state (recommended value between -1 and 1, but can be larger than these values). 

#### Returns:

* predictedValue: A value that is predicted by the model.

* probability: The probability of predicted value.

### reset()

Reset model's stored values (excluding the parameters).

```
StateActionRewardStateActionNeuralNetwork:reset()
```

## Inherited From

* [NeuralNetwork](NeuralNetwork.md)

## References

* [Reinforcement Learning with Neural Network](https://www.baeldung.com/cs/reinforcement-learning-neural-network)

* [Deep Q Networks (DQN) in Python From Scratch by Using OpenAI Gym and TensorFlow- Reinforcement Learning Tutorial](https://aleksandarhaber.com/deep-q-networks-dqn-in-python-from-scratch-by-using-openai-gym-and-tensorflow-reinforcement-learning-tutorial/)

* [Reinforcement Q-Learning from Scratch in Python with OpenAI Gym](https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/)
