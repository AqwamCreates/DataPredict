# [API Reference](../../API.md) - [Models](../Models.md) - GenerativeAdversarialNetwork (GAN)

GenerativeAdversarialNetwork uses two neural networks to generate new content from noise.

## Notes:

* The GeneratorNeuralNetwork and DiscriminatorNeuralNetwork models must be created separately. Then use setGeneratorNeuralNetwork() and setDiscriminatorNeuralNetwork() to put it inside the GenerativeAdversarialNetwork model.

* GeneratorNeuralNetwork and DiscriminatorNeuralNetwork models must be a part of NeuralNetwork model. If you decide to use linear regression or logistic regression, then it must be constructed using NeuralNetwork model. 

* Ensure the final layer of the DiscriminatorNeuralNetwork model has only one neuron. It is the default setting for all DiscriminatorNeuralNetwork models in research papers.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
GenerativeAdversarialNetwork.new(maxNumberOfIterations: number): ModelObject
```

#### Parameters:

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
GenerativeAdversarialNetwork:setParameters(maxNumberOfIterations: number)
```

#### Parameters:

* discountFactor: The higher the value, the more likely it focuses on long-term outcomes. The value must be set between 0 and 1.

## References

* [Proximal Policy Optimization By OpenAI](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
