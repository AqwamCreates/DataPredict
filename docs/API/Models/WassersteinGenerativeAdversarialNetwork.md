# [API Reference](../../API.md) - [Models](../Models.md) - WassersteinGenerativeAdversarialNetwork (GAN)

WassersteinGenerativeAdversarialNetwork uses two neural networks to generate new contents from noise.

## Notes

* The Generator and Discriminator models must be created separately. Then use setGeneratorModel() and setDiscriminatorModel() to put it inside the WassersteinGenerativeAdversarialNetwork model.

* Generator and Discriminator models must be a part of NeuralNetwork model. If you decide to use linear regression or logistic regression, then it must be constructed using NeuralNetwork model. 

* Ensure the output layer of the Discriminator model has only one neuron and its activation function set to "LeakyReLU". It is the default setting for all Discriminator models in research papers.

* The number of neurons at the Generator's output layer must be equal to the number of neurons at the Discriminator's input layer.

* It is recommended that the learning rate for the Generator is higher than the Discriminator.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
WassersteinGenerativeAdversarialNetwork.new(generatorMaximumNumberOfIterations: number, discriminatorMaximumNumberOfIterations: number, sampleSize: integer): ModelObject
```

#### Parameters:

* generatorMaximumNumberOfIterations: The number of times that the generator should be trained.

* discriminatorMaximumNumberOfIterations: The number of times that the discriminator should be trained for each of generator's iteration.

* sampleSize: How many randomly chosen data will be used from the real feature matrix and noise feature matrix on every iteration.

#### Returns:

* ModelObject: The generated model object.

## Functions

### train()

Trains the model.

```
WassersteinGenerativeAdversarialNetwork:train(realFeatureMatrix: matrix, noiseFeatureMatrix: matrix)
```

#### Parameters:

* realFeatureMatrix: The matrix containing the features of the real contents. The number of columns must be equal to number of neurons at the output layer of the Generator.

* noiseFeatureMatrix: The matrix containing the noise in order to generate fake contents. The number of columns must be equal to number of neurons at the input layer of the Generator.

### evaluate()

Generates the output from Discriminator.

```
WassersteinGenerativeAdversarialNetwork:evaluate(featureMatrix: matrix): matrix
```

#### Parameters:

* featureMatrix: The matrix containing all data.

#### Returns:

* outputMatrix: The matrix containing all the output values.

### generate()

Generates the output from Generator.

```
WassersteinGenerativeAdversarialNetwork:generate(noiseFeatureMatrix: matrix): matrix
```

#### Parameters:

* noiseFeatureMatrix: The matrix containing the noise in order to generate fake contents. The number of columns must be equal to number of neurons at the input layer of the Generator.

#### Returns:

* outputMatrix: Matrix containing all the output values.

## Inherited From

* [GenerativeAdversarialNetworkBaseModel](GenerativeAdversarialNetworkBaseModel.md)

## References

* [Wasserstein GAN](https://arxiv.org/abs/1701.07875)

* [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
