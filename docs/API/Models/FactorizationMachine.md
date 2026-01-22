# [API Reference](../../API.md) - [Models](../Models.md) - FactorizationMachine

FactorizationMachine is a supervised machine learning model that predicts continuous values (e.g. 1.2, -32, 90, -1.2 and etc. ). It uses iterative calculations to find the best model parameters.

Can be converted into classification.

## Stored Model Parameters

Contains a table of matrices.  

* ModelParameters[1][I][J]: weightMatrix, Value of matrix at row I and column J. The rows are the features.

* ModelParameters[2][I][J]: latentWeightMatrix, Value of matrix at row I and column J. The rows are the features and the columns are the latent factors.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
FactorizationMachine.new(maximumNumberOfIterations: integer, learningRate: number, binaryFunction: string, costFunction: string): ModelObject
```

#### Parameters:

* maximumNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between 0 to 1.

* binaryFunction: The binary function to be used by the model. Available options are:

| Function            | Output Range | Skewness              | Use Cases                                                                  |
|---------------------|--------------|-----------------------|----------------------------------------------------------------------------|
| None (Default)      | (-∞, ∞)      | Symmetric             | Rating Prediction                                                          |
| Logistic            | (0, 1)       | Symmetric             | Player Choice (A/B), Engagement Prediction, Click-Through Rates            |
| HardSigmoid         | (0, 1)       | Symmetric             | Same As Logistic, But Mobile / Real-Time Prediction                        |
| Probit              | (0, 1)       | Symmetric             | Skill-Based Success, Ability Checks, Normally Distributed Traits           |
| BipolarSigmoid      | (-1, 1)      | Symmetric             | Win / Lose, Accept / Reject, Binary Outcomes With Magnitude                |
| Tanh                | (-1, 1)      | Symmetric             | Like / Dislike, Positive / Negative Feedback, Preference Modeling          |
| SoftSign            | (-1, 1)      | Symmetric             | Gradual Preference Changes, Soft Decisions                                 |
| ArcTangent          | (-π/2, π/2)  | Symmetric             | Academic / Research Alternative To Tanh                                    |
| ComplementaryLogLog | (0, 1)       | Right-Skewed          | Rare Events Prediction: In-App Purchases, Time-To-Leave Prediction         |
| LogLog              | (0, 1)       | Left-Skewed           | Common Events Prediction: Tutorial Completion, Early Wins, First Purchases |

* costFunction: The function to calculate the cost of each training. Available options are: 

  * MeanSquaredError (Default)

  * MeanAbsoluteError

  * BinaryCrossEntropy
 
  * HingeLoss

#### Returns:

* ModelObject: The generated model object.

## Functions

### setOptimizer()

Set optimizer for the model by inputting the optimizer object.

```
FactorizationMachine:setOptimizer(Optimizer: OptimizerObject)
```

#### Parameters:

* Optimizer: The optimizer object to be used.

### setRegularizer()

Set a regularization for the model by inputting the optimizer object.

```
FactorizationMachine:setRegularizer(Regularizer: RegularizerObject)
```

#### Parameters:

* setRegularizer: The regularizer to be used.

### train()

Train the model.

```
FactorizationMachine:train(featureMatrix: Matrix, labelVector: Matrix): number[]
```

#### Parameters:

* featureMatrix: Matrix containing all data.

* labelVector: A (n x 1) matrix containing values related to featureMatrix.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict the value for a given data.

```
FactorizationMachine:predict(featureMatrix: Matrix): Matrix
```

#### Parameters:

* featureMatrix: Matrix containing data.

#### Returns:

* predictedVector: A vector containing values that are predicted by the model.

## Inherited From

* [GradientMethodBaseModel](GradientMethodBaseModel.md)

## References

* [Factorization Machines](https://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle2010FM.pdf)
