# [API Reference](../../API.md) - [Models](../Models.md) - GenerativeAdversarialNetwork (GAN)

GenerativeAdversarialNetwork uses two neural networks to generate new content from noise.

## Notes:

* The Generator and Discriminator models must be created separately. Then use setGenerator() and setDiscriminator() to put it inside the GenerativeAdversarialNetwork model.

* Generator and Discriminator models must be a part of NeuralNetwork model. If you decide to use linear regression or logistic regression, then it must be constructed using NeuralNetwork model. 

* Ensure the final layer of the Discriminator model has only one neuron and its activation function set to "Sigmoid". It is the default setting for all Discriminator models in research papers.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
GenerativeAdversarialNetwork.new(maxNumberOfIterations: number): ModelObject
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
GenerativeAdversarialNetwork:setParameters(maxNumberOfIterations: number)
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

### setGenerator()

Sets the Generator into the model. 

```
GenerativeAdversarialNetwork:setGenerator(Generator: Model)
```

#### Parameters:

* Generator: The model to be used for generating contents out of random noise.

### setDiscriminator()

Sets the Discriminator into the model. 

```
GenerativeAdversarialNetwork:setDiscriminator(Discriminator: Model)
```

#### Parameters:

* Discriminator: The model to be used for discriminating real and fake contents.

### train()

Trains the model.

```
GenerativeAdversarialNetwork:train(realFeatureMatrix: matrix, noiseFeatureMatrix: matrix)
```

#### Parameters:

* realFeatureMatrix: The matrix containing the features of the real contents.

* noiseFeatureMatrix: The matrix containing the noise in order to generate fake contents.

### evaluate()

Generates the output from Discriminator.

```
GenerativeAdversarialNetwork:evaluate(featureMatrix: matrix): matrix
```

#### Parameters:

* featureMatrix: The matrix containing the features of the real contents.

#### Returns:

* outputMatrix: The matrix containing all the output values.

### generate()

Generates the output from Generator.

```
GenerativeAdversarialNetwork:evaluate(featureMatrix: matrix): matrix
```

#### Parameters:

* featureMatrix: The matrix containing the features of the real contents.

#### Returns:

* outputMatrix: The matrix containing all the output values.

## References

* [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
