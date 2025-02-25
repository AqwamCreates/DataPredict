# [API Reference](../../API.md) - [ReinforcementLearningStrategies](../ReinforcementLearningStrategies.md) - GenerativeAdversarialImitationLearningBaseModel

WassersteinGenerativeAdversarialImitationLearning allows an agent to learn from expert's trajectories.

## Notes

* The ReinforcementLearning and Discriminator models must be created separately. Then use setReinforcementLearningModel() and setDiscriminatorModel() to put it inside the GenerativeAdversarialImitationLearning model.

* ReinforcementLearning and Discriminator models must be a part of NeuralNetwork model. If you decide to use linear regression or logistic regression, then it must be constructed using NeuralNetwork model. 

* Ensure the output layer of the Discriminator model has only one neuron and its activation function set to "Sigmoid". It is the default setting for all Discriminator models in research papers.

* The number of neurons at the ReinforcementLearning's output layer must be equal to the number of neurons at the Discriminator's input layer.

* It is recommended that the learning rate for the ReinforcementLearning is higher than the Discriminator.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
WassersteinGenerativeAdversarialImitationLearning.new(numberOfStepsPerEpisode: number): ModelObject
```

#### Parameters:

* numberOfStepsPerEpisode: How many steps are needed for it to be considered as a single episode.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setReinforcementLearningModel()

Sets the ReinforcementLearning into the model. 

```
WassersteinGenerativeAdversarialImitationLearning:setReinforcementLearningModel(ReinforcementLearningModel: Model)
```

#### Parameters:

* ReinforcementLearningModel: The model to be used for mimicking the expert.

### setDiscriminatorModel()

Sets the Discriminator into the model. 

```
WassersteinGenerativeAdversarialImitationLearning:setDiscriminatorModel(DiscriminatorModel: Model)
```

#### Parameters:

* Discriminator: The model to be used for discriminating real and fake contents.

### getReinforcementLearningModel()

Gets the ReinforcementLearning from the model. 

```
WassersteinGenerativeAdversarialImitationLearning:getReinforcementLearningModel(): Model
```

#### Returns:

* GeneratorModel: The model used for generating contents out of random noise.

### getDiscriminatorModel()

Gets the Discriminator from the model. 

```
WassersteinGenerativeAdversarialImitationLearning:getDiscriminatorModel(): Model
```

#### Returns:

* DiscriminatorModel: The model used for discriminating real and fake contents.

### setClassesList()

```
OneVsAll:setClassesList(ClassesList: [])
```

#### Parameters:

* ClassesList: A list of classes. The index of the list relates to which model belong to. For example, {3, 1} means that the output for 3 is at first model, and the output for 1 is at second model.

### getClassesList()

```
OneVsAll:getClassesList(): []
```

#### Returns:

* ClassesList: A list of classes. The index of the list relates to which model belong to. For example, {3, 1} means that the output for 3 is at first model, and the output for 1 is at second model.

## Inherited From

* [BaseInstance](BaseInstance.md)

## References

* [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476)

* [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
