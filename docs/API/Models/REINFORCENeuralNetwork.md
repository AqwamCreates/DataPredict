# [API Reference](../../API.md) - [Models](../Models.md) - REINFORCENeuralNetwork

REINFORCE is a neural network with reinforcement learning capabilities. It can predict any positive numbers of discrete values.

## Stored Model Parameters

Contains a table of matrices.  

* ModelParameters[L][I][J]: Matrix at layer L. Value of matrix at row I and column J.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
REINFORCENeuralNetwork.new(maxNumberOfIterations: integer, learningRate: number, discountFactor: number): ModelObject
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
REINFORCENeuralNetwork:setParameters(maxNumberOfIterations: integer, learningRate: number, discountFactor: number)
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

## Inherited From

* [ReinforcementLearningNeuralNetworkBaseModel](ReinforcementLearningNeuralNetworkBaseModel.md)

## References

* [REINFORCE – A Quick Introduction (with Code)](https://dilithjay.com/blog/reinforce-a-quick-introduction-with-code/)

* [REINFORCE — a policy-gradient based reinforcement Learning algorithm](https://medium.com/intro-to-artificial-intelligence/reinforce-a-policy-gradient-based-reinforcement-learning-algorithm-84bde440c816)

* [REINFORCE](https://paperswithcode.com/method/reinforce)
