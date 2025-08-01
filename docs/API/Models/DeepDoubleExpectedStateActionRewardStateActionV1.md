# [API Reference](../../API.md) - [Models](../Models.md) - DeepDoubleExpectedStateActionRewardStateActionV1 (Double Deep Expected SARSA)

DeepDoubleExpectedStateActionRewardStateActionV1 is a neural network with reinforcement learning capabilities. It can predict any positive numbers of discrete values.

## Stored Model Parameters

Contains a table of matrices.  

* ModelParameters[L][I][J]: Matrix at layer L. Value of matrix at row I and column J.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
DeepDoubleExpectedStateActionRewardStateAction.new(epsilon: number, lambda: number, discountFactor: number): ModelObject
```

#### Parameters:

* epsilon: Controls the balance between exploration and exploitation for calculating expected Q-values. The value must be set between 0 and 1. The value 0 focuses on exploitation only and 1 focuses on exploration only. [Default: 0.5]

* lambda: At 0, the model acts like the Temporal Difference algorithm. At 1, the model acts as Monte Carlo algorithm. Between 0 and 1, the model acts as both. [Default: 0]

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1. [Default: 0.95]

#### Returns:

* ModelObject: The generated model object.

## Functions

### setModelParameters1()

Sets model parameters to be used by the model.

```
DeepDoubleExpectedStateActionRewardStateAction:setModelParameters1(ModelParameters1: ModelParameters)
```

#### Parameters:

* ModelParameters1: First model parameters to be used by the model.

### setModelParameters2()

Sets model parameters to be used by the model.

```
DeepDoubleExpectedStateActionRewardStateAction:setModelParameters1(ModelParameters2: ModelParameters)
```

#### Parameters:

* ModelParameters2: Second model parameters to be used by the model.

### setModelParameters1()

Sets model parameters to be used by the model.

```
DeepDoubleExpectedStateActionRewardStateAction:setModelParameters1(ModelParameters1: ModelParameters, doNotDeepCopy: boolean)
```

#### Parameters:

* ModelParameters1: First model parameters to be used by the model.

* doNotDeepCopy: Set whether or not to deep copy the model parameters.

### setModelParameters2()

Sets model parameters to be used by the model.

```
DeepDoubleExpectedStateActionRewardStateAction:setModelParameters2(ModelParameters2: ModelParameters, doNotDeepCopy: boolean)
```

#### Parameters:

* ModelParameters2: Second model parameters to be used by the model.

* doNotDeepCopy: Set whether or not to deep copy the model parameters.


### getModelParameters1()

Sets model parameters to be used by the model.

```
DeepDoubleExpectedStateActionRewardStateAction:getModelParameters1(doNotDeepCopy: boolean): ModelParameters
```

#### Parameters:

* doNotDeepCopy: Set whether or not to deep copy the model parameters.

#### Returns:

* ModelParameters1: First model parameters that was used by the model.

### getModelParameters2()

Sets model parameters to be used by the model.

```
DeepDoubleExpectedStateActionRewardStateAction:getModelParameters2(doNotDeepCopy: boolean): ModelParameters
```

#### Parameters:

* doNotDeepCopy: Set whether or not to deep copy the model parameters.

#### Returns:

* ModelParameters2: Second model parameters that was used by the model.

## Inherited From

* [ReinforcementLearningBaseModel](ReinforcementLearningBaseModel.md)

## References

* [Double Sarsa and Double Expected Sarsa with Shallow and Deep Learning](https://www.scirp.org/journal/paperinformation.aspx?paperid=71237)

* [SARSA Reinforcement Learning](https://www.geeksforgeeks.org/sarsa-reinforcement-learning/)

* [Reinforcement Learning with Neural Network](https://www.baeldung.com/cs/reinforcement-learning-neural-network)
