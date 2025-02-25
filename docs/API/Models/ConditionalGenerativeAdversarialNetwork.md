# [API Reference](../../API.md) - [Models](../Models.md) - ConditionalGenerativeAdversarialNetwork (CGAN)

ConditionalGenerativeAdversarialNetwork uses two neural networks to generate new contents from noise.

## Notes

* The Generator and Discriminator models must be created separately. Then use setGeneratorModel() and setDiscriminatorModel() to put it inside the ConditionalGenerativeAdversarialNetwork model.

* Generator and Discriminator models must be a part of NeuralNetwork model. If you decide to use linear regression or logistic regression, then it must be constructed using NeuralNetwork model. 

* Ensure the output layer of the Discriminator model has only one neuron and its activation function set to "Sigmoid". It is the default setting for all Discriminator models in research papers.

* The number of neurons at the Generator's input layer must be equal to the total the number of neurons at the Discriminator's output layer and the number of features in label matrix.

* It is recommended that the learning rate for the Generator is higher than the Discriminator.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
ConditionalGenerativeAdversarialNetwork.new(maximumNumberOfIterations: number): ModelObject
```

#### Parameters:

* maximumNumberOfIterations: How many times should the model needed to be trained.

#### Returns:

* ModelObject: The generated model object.

## Functions

### train()

Trains the model.

```
ConditionalGenerativeAdversarialNetwork:train(realFeatureMatrix: matrix, noiseFeatureMatrix: matrix, labelMatrix: matrix)
```

#### Parameters:

* realFeatureMatrix: The matrix containing the features of the real contents. The number of columns must be equal to number of neurons at the output layer of the Generator.

* noiseFeatureMatrix: The matrix containing the noise in order to generate fake contents. The number of columns must be equal to number of neurons at the input layer of the Generator.

* labelMatrix: The matrix containing the class labels corresponding to the real feature matrix.

### evaluate()

Generates the output from Discriminator.

```
ConditionalGenerativeAdversarialNetwork:evaluate(featureMatrix: matrix, labelMatrix: matrix): matrix
```

#### Parameters:

* featureMatrix: The matrix containing all data.

* labelMatrix: The matrix containing the class labels corresponding to the feature matrix.

#### Returns:

* outputMatrix: The matrix containing all the output values.

### generate()

Generates the output from Generator.

```
ConditionalGenerativeAdversarialNetwork:generate(noiseFeatureMatrix: matrix, labelMatrix: matrix): matrix
```

#### Parameters:

* noiseFeatureMatrix: The matrix containing the noise in order to generate fake contents. The number of columns must be equal to number of neurons at the input layer of the Generator.

* labelMatrix: The matrix containing the class labels corresponding to the real feature matrix.

#### Returns:

* outputMatrix: Matrix containing all the output values.

## Inherited From

* [GenerativeAdversarialNetworkBaseModel](GenerativeAdversarialNetworkBaseModel.md)

## References

* [ACGAN Architectural Design](https://stephan-osterburg.gitbook.io/coding/coding/ml-dl/tensorfow/chapter-4-conditional-generative-adversarial-network/acgan-architectural-design)

* [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
