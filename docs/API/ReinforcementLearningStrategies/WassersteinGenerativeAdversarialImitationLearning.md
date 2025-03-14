# [API Reference](../../API.md) - [ReinforcementLearningStrategies](../ReinforcementLearningStrategies.md) - WassersteinGenerativeAdversarialImitationLearning (WGAIL)

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

### categoricalTrain()

Categorically trains the model.

```
WassersteinGenerativeAdversarialImitationLearning:categoricalTrain(previousFeatureMatrix: matrix, expertActionMatrix: matrix, currentFeatureMatrix: matrix, terminalStateMatrix: matrix)
```

#### Parameters:

* previousFeatureMatrix: The matrix containing the feature environment values.

* expertActionMatrix: The matrix containing the action values generated by the expert.

* currentFeatureMatrix: The matrix containing the feature environment values after an action has been taken by the expert.

* terminalStateMatrix: The matrix containing the terminal state values that have been encountered by the expert.

### diagonalGaussianTrain()

Diagonally Gaussian trains the model.

```
WassersteinGenerativeAdversarialImitationLearning:diagonalGaussianTrain(previousFeatureMatrix: matrix, expertActionMeanMatrix: matrix, expertStandardDeviationMatrix: matrix, expertActionNoiseMatrix: matrix, currentFeatureMatrix: matrix, terminalStateMatrix: matrix)
```

#### Parameters:

* previousFeatureMatrix: The matrix containing the feature environment values.

* expertActionMeanMatrix: The matrix containing the action mean values generated by the expert.
  
* expertStandardDeviationMatrix: The matrix containing the action standard deviation values generated by the expert.

* expertActionNoiseMatrix: The matrix containing the action noise values generated by the expert.

* currentFeatureMatrix: The matrix containing the feature environment values after the action has been taken by the expert.

* terminalStateMatrix: The matrix containing the terminal state values that have been encountered by the expert.

### evaluate()

Generates the output from Discriminator.

```
WassersteinGenerativeAdversarialImitationLearning:evaluate(featureMatrix: matrix): matrix
```

#### Parameters:

* featureMatrix: The matrix containing the feature environment values.

#### Returns:

* outputMatrix: The matrix containing all the output values.

### generate()

Generates the output from Generator.

```
WassersteinGenerativeAdversarialImitationLearning:generate(featureMatrix: matrix, returnOriginalOutput: boolean): matrix 
```

#### Parameters:

* featureMatrix: The matrix containing the feature environment values.

#### Returns:

* actionVector: The vector containing the actions with the highest values.

-OR-

* actionMatrix: The matrix containing all the action values.

## Inherited From

* [GenerativeAdversarialImitationLearningBaseModel](GenerativeAdversarialImitationLearningBaseModel.md)

## References

* [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476)

* [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
