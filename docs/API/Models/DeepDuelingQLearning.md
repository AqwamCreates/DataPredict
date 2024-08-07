# [API Reference](../../API.md) - [Models](../Models.md) - DeepDuelingQLearning

DeepDuelingQLearning is a base class for reinforcement learning.

## Notes:

* The Advantage and Value models must be created separately. Then use setAdvantageModel() and setValueModel() to put it inside the DeepDuelingQLearning model.

* Advantage and Value models must be a part of NeuralNetwork model. If you decide to use linear regression or logistic regression, then it must be constructed using NeuralNetwork model. 

* Ensure the final layer of the Value model has only one neuron. It is the default setting for all Value models in research papers.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
DeepDuelingQLearning.new(discountFactor: number): ModelObject
```

#### Parameters:

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
DeepDuelingQLearning:setParameters(discountFactor: number)
```

#### Parameters:

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

## Inherited From

* [ReinforcementLearningDeepDuelingQLearningBaseModel](ReinforcementLearningDeepDuelingQLearningBaseModel.md)

## References

* [Dueling Deep Q Networks by Chris Yoon](https://towardsdatascience.com/dueling-deep-q-networks-81ffab672751)
