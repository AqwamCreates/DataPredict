# [API Reference](../../API.md) - [Models](../Models.md) - GenerativeAdversarialNetworkBaseModel

GenerativeAdversarialNetwork uses two neural networks to generate new contents from noise.

## Notes

* The Generator and Discriminator models must be created separately. Then use setGeneratorModel() and setDiscriminatorModel() to put it inside the GenerativeAdversarialNetwork model.

* Generator and Discriminator models must be a part of NeuralNetwork model. If you decide to use linear regression or logistic regression, then it must be constructed using NeuralNetwork model. 

* Ensure the output layer of the Discriminator model has only one neuron and its activation function set to "Sigmoid". It is the default setting for all Discriminator models in research papers.

* The number of neurons at the Generator's output layer must be equal to the number of neurons at the Discriminator's input layer.

* It is recommended that the learning rate for the Generator is higher than the Discriminator.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
GenerativeAdversarialNetwork.new(maximumNumberOfIterations: number): ModelObject
```

#### Parameters:

* maximumNumberOfIterations: How many times should the model needed to be trained.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setGeneratorModel()

Sets the Generator into the model. 

```
GenerativeAdversarialNetwork:setGeneratorModel(GeneratorModel: Model)
```

#### Parameters:

* Generator: The model to be used for generating contents out of random noise.

### setDiscriminatorModel()

Sets the Discriminator into the model. 

```
GenerativeAdversarialNetwork:setDiscriminatorModel(DiscriminatorModel: Model)
```

#### Parameters:

* Discriminator: The model to be used for discriminating real and fake contents.

### getGeneratorModel()

Gets the Generator from the model. 

```
GenerativeAdversarialNetwork:getGeneratorModel(): Model
```

#### Returns:

* GeneratorModel: The model used for generating contents out of random noise.

### getDiscriminatorModel()

Gets the Discriminator from the model. 

```
GenerativeAdversarialNetwork:getDiscriminatorModel(): Model
```

#### Returns:

* DiscriminatorModel: The model used for discriminating real and fake contents.

#### Parameters:

* realFeatureMatrix: The matrix containing the features of the real contents. The number of columns must be equal to number of neurons at the output layer of the Generator.

* noiseFeatureMatrix: The matrix containing the noise in order to generate fake contents. The number of columns must be equal to number of neurons at the input layer of the Generator.

#### Parameters:

* noiseFeatureMatrix: The matrix containing the noise in order to generate fake contents. The number of columns must be equal to number of neurons at the input layer of the Generator.

#### Returns:

* outputMatrix: Matrix containing all the output values.

## Inherited From

* [IterativeMethodBaseModel](IterativeMethodBaseModel.md.md)

## References

* [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
