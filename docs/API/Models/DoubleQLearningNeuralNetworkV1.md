# [API Reference](../../API.md) - [Models](../Models.md) - DoubleQLearningNeuralNetworkV1 (DDQN / Double Deep Q-Learning)

DoubleQLearningNeuralNetworkV1 is a neural network with reinforcement learning capabilities. It can predict any positive numbers of discrete values.

It uses Hasselt et al. (2010) version, where a single neural network is selected from two neural networks with equal probability for training.

## Stored Model Parameters

Contains a table of matrices.  

* ModelParameters[L][I][J]: Matrix at layer L. Value of matrix at row I and column J.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
DoubleQLearningNeuralNetwork.new(maxNumberOfIterations: integer discountFactor: number): ModelObject
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
DoubleQLearningNeuralNetwork:setParameters(maxNumberOfIterations: integer, discountFactor: number)
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

### setModelParameters1()

Sets model parameters to be used by the model.

```
DoubleQLearningNeuralNetwork:setModelParameters1(ModelParameters1: ModelParameters)
```

#### Parameters:

* ModelParameters1: First model parameters to be used by the model.

### setModelParameters2()

Sets model parameters to be used by the model.

```
DoubleQLearningNeuralNetwork:setModelParameters1(ModelParameters2: ModelParameters)
```

#### Parameters:

* ModelParameters2: Second model parameters to be used by the model.

### getModelParameters1()

Sets model parameters to be used by the model.

```
DoubleQLearningNeuralNetwork:getModelParameters1(): ModelParameters
```

#### Returns:

* ModelParameters1: First model parameters that was used by the model.

### getModelParameters2()

Sets model parameters to be used by the model.

```
DoubleQLearningNeuralNetwork:getModelParameters2(): ModelParameters
```

#### Returns:

* ModelParameters2: Second model parameters that was used by the model.

## Inherited From

* [ReinforcementLearningNeuralNetworkBaseModel](ReinforcementLearningNeuralNetworkBaseModel.md)

## References

* [Double Deep Q Networks](https://towardsdatascience.com/double-deep-q-networks-905dd8325412)

* [Reinforcement Learning with Neural Network](https://www.baeldung.com/cs/reinforcement-learning-neural-network)

* [Deep Q Networks (DQN) in Python From Scratch by Using OpenAI Gym and TensorFlow- Reinforcement Learning Tutorial](https://aleksandarhaber.com/deep-q-networks-dqn-in-python-from-scratch-by-using-openai-gym-and-tensorflow-reinforcement-learning-tutorial/)

* [Reinforcement Q-Learning from Scratch in Python with OpenAI Gym](https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/)
