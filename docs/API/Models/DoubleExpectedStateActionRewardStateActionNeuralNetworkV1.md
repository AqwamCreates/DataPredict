# [API Reference](../../API.md) - [Models](../Models.md) - DoubleExpectedStateActionRewardStateActionNeuralNetworkV1 (Double Deep Expected SARSA)

DoubleExpectedStateActionRewardStateActionNeuralNetworkV1 is a neural network with reinforcement learning capabilities. It can predict any positive numbers of discrete values.

## Stored Model Parameters

Contains a table of matrices.  

* ModelParameters[L][I][J]: Matrix at layer L. Value of matrix at row I and column J.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
DoubleExpectedStateActionRewardStateActionNeuralNetworkV1.new(maxNumberOfIterations: integer, learningRate: number, epsilon: number, discountFactor: number): ModelObject
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* epsilon: Controls the balance between exploration and exploitation for calculating expected q values. The value must be set between 0 and 1. The value 0 focuses on exploitation only and 1 focuses on exploration only.

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
DoubleExpectedStateActionRewardStateActionNeuralNetworkV1:setParameters(maxNumberOfIterations: integer, learningRate: number, epsilon: number, discountFactor: number)
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* epsilon: Controls the balance between exploration and exploitation for calculating expected q values. The value must be set between 0 and 1. The value 0 focuses on exploitation only and 1 focuses on exploration only.

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

### setModelParameters1()

Sets model parameters to be used by the model.

```
DoubleExpectedStateActionRewardStateActionNeuralNetworkV1:setModelParameters1(ModelParameters1: ModelParameters)
```

#### Parameters:

* ModelParameters1: First model parameters to be used by the model.

### setModelParameters2()

Sets model parameters to be used by the model.

```
DoubleExpectedStateActionRewardStateActionNeuralNetworkV1:setModelParameters1(ModelParameters2: ModelParameters)
```

#### Parameters:

* ModelParameters2: Second model parameters to be used by the model.

### getModelParameters1()

Sets model parameters to be used by the model.

```
DoubleExpectedStateActionRewardStateActionNeuralNetworkV1:getModelParameters1(): ModelParameters
```

#### Returns:

* ModelParameters1: First model parameters that was used by the model.

### getModelParameters2()

Sets model parameters to be used by the model.

```
DoubleExpectedStateActionRewardStateActionNeuralNetworkV1:getModelParameters2(): ModelParameters
```

#### Returns:

* ModelParameters2: Second model parameters that was used by the model.

## Inherited From

* [ReinforcementLearningNeuralNetworkBaseModel](ReinforcementLearningNeuralNetworkBaseModel.md)

## References

* [Double Sarsa and Double Expected Sarsa with Shallow and Deep Learning](https://www.scirp.org/journal/paperinformation.aspx?paperid=71237)

* [Expected SARSA in Reinforcement Learning](https://www.geeksforgeeks.org/expected-sarsa-in-reinforcement-learning/)

* [SARSA Reinforcement Learning](https://www.geeksforgeeks.org/sarsa-reinforcement-learning/)

* [Reinforcement Learning with Neural Network](https://www.baeldung.com/cs/reinforcement-learning-neural-network)

* [Deep Q Networks (DQN) in Python From Scratch by Using OpenAI Gym and TensorFlow- Reinforcement Learning Tutorial](https://aleksandarhaber.com/deep-q-networks-dqn-in-python-from-scratch-by-using-openai-gym-and-tensorflow-reinforcement-learning-tutorial/)

* [Reinforcement Q-Learning from Scratch in Python with OpenAI Gym](https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/)
