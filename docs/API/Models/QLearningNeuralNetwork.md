# [API Reference](../../API.md) - [Models](../Models.md) - QLearningNeuralNetwork

NeuralNetwork is a supervised machine learning model that predicts any positive numbers of discrete values.

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

* epsilon: The higher the value, more likely it focuses on exploration over exploitation.

* epsilonDecayFactor: Controls how fast the model goes from exploring to exploiting as it learns.

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

* epsilon: The higher the value, more likely it focuses on exploration over exploitation.

* epsilonDecayFactor: Controls how fast the model goes from exploring to exploiting as it learns.

* discountFactor: Balances present and future rewards in agent decisions.

### reinforce()

Reward or punish model based on the predicted output.

```
QLearningNeuralNetwork:reinforce(featureVector: Matrix, rewardValue: number): integer, number
```

#### Parameters:

* featureVector: Matrix containing data.

* rewardValue: How much do we reward the model if it gets the prediction correct (value between 0 and 1).

#### Returns:

* predictedValue: A value that is predicted by the model.

* probability: The probability of predicted value.

## Inherited From

* [NeuralNetwork](NeuralNetwork.md)
