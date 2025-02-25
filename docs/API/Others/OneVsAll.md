# [API Reference](../../API.md) - [Others](../Others.md) - OneVsAll

Allows binary classification models (such as LogisticRegression) be merged together to form multi-class models.

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
OneVsAll.new(maxNumberOfIterations: integer, useNegativeOneBinaryLabel: boolean): OneVsAllObject
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* useNegativeOneBinaryLabel: Set whether or not if the negative labels uses -1 instead of 0.

#### Returns:

* OneVsAllObject: The generated OneVsAll object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
OneVsAll:setParameters(maxNumberOfIterations: integer, useNegativeOneBinaryLabel: boolean)
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* useNegativeOneBinaryLabel: Set whether or not if the negative labels uses -1 instead of 0.

### setModels()

Sets the model and number of classes to be used by the OneVsAll object. Leaving it empty will clear the model.

```
OneVsAll:setModels(modelName: string, numberOfClasses: integer)
```

#### Parameters:

* modelName: The full name of the model to be used in OneVsAll object.

* numberOfClasses: The number of models to be generated based on number of classes.

### setOptimizer()

Sets the optimizer and its parameters. Leaving it empty will clear the optimizer.

```
OneVsAll:setOptimizer(optimizerName: string, ...)
```

#### Parameters:

* optimizerName: The full name of the optimizer to be used in OneVsAll object.

* ...: The parameters to be provided to the optimizer. 

### setRegularizer()

Sets the regularizer and its parameters. Leaving it empty will clear the optimizer.

```
OneVsAll:setRegularizer(lambda: number, regularizationMode: string, hasBias: boolean)
```

#### Parameters:

* lambda: Regularization factor. Recommended values are between 0 to 1.

* regularisationMode: The mode which regularization will be used. Currently available ones are "L1" (or "Lasso"), "L2" (or "Ridge") and "L1+L2" (or "ElasticNet").

* hasBias: Set whether or not the regularization has bias.

### setModelsSettings()

```
OneVsAll:setModelsSettings(...: any)
```

#### Parameters:

* ...: The parameters to be set to all models stored in this OneVsAll object.

### train()

Train the model.

```
NeuralNetwork:train(featureMatrix: Matrix, labelVector / labelMatrix: Matrix): number[]
```
#### Parameters:

* featureMatrix: Matrix containing all data.

* labelVector / labelMatrix: A (n x 1) / (n x o) matrix containing values related to featureMatrix. When using the label matrix, the number of columns must be equal to number of classes.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict the values for given data.

```
OneVsAll:predict(featureMatrix: Matrix): Matrix, Matrix
```

#### Parameters:

* featureMatrix: Matrix containing all data.

* returnOriginalOutput: Set whether or not to return predicted matrix instead of value with highest probability. 

#### Returns:

* predictedVector: A vector that is predicted by the model.

* highestValueVector: A vector that contains the predicted values in predictedVector.

### getClassesList()

```
OneVsAll:getClassesList(): []
```

#### Returns:

* ClassesList: A list of classes. The index of the list relates to which model belong to. For example, {3, 1} means that the output for 3 is at first model, and the output for 1 is at second model.

### setClassesList()

```
OneVsAll:setClassesList(ClassesList: [])
```

#### Parameters:

* ClassesList: A list of classes. The index of the list relates to which model belong to. For example, {3, 1} means that the output for 3 is at first model, and the output for 1 is at second model.

### getModelParametersArray()

Gets the model parameters from the base model.

```
OneVsAll:getModelParametersArray(doNotDeepCopy: boolean): ModelParameters []
```

#### Parameters

* doNotDeepCopy: Set whether or not to deep copy the model parameters.

#### Returns

* ModelParameters: An array containing model parameters (matrix/table) fetched from each model. The index of the array determines which model it belongs to.

### setModelParametersArray()

Set the model parameters to the base model.

```
OneVsAll:setModelParameters(ModelParametersArray: ModelParameters[], doNotDeepCopy: boolean)
```

#### Parameters

* ModelParametersArray: A table containing model parameters (matrix/table) to be given to be given to each model stored in OneVsAll object.  The position of the parameters determines which model it belongs to.

* doNotDeepCopy: Set whether or not to deep copy the model parameters.

### clearModelParameters()

Clears the model parameters stored inside the models.

```
OneVsAll:clearModelParameters()
```

### setNumberOfIterationsToCheckIfConverged()

Set the number of iterations needed to confirm convergence for each model.

```
OneVsAll:setNumberOfIterationsToCheckIfConverged(numberOfIterations: number)
```

#### Parameters

* numberOfIterations: The number of iterations for confirming convergence.

### setNumberOfIterationsToCheckIfConvergedForOneVsAll()

Set the number of iterations needed to confirm convergence.

```
OneVsAll:setNumberOfIterationsToCheckIfConvergedForOneVsAll(numberOfIterations: number)
```

#### Parameters

* numberOfIterations: The number of iterations for confirming convergence.

### setTargetCost()

Set the upper bound and lower bounds of the target cost for each model.
```
OneVsAll:setTargetCost(upperBound: number, lowerBound: number)
```

#### Parameters

* upperBound: The upper bound of target cost.

* lowerBound: The lower bound of target cost.

### setTargetTotalCost()

Set the upper bound and lower bounds of the target cost.
```
OneVsAll:setTargetTotalCost(upperBound: number, lowerBound: number)
```

#### Parameters

* upperBound: The upper bound of target cost.

* lowerBound: The lower bound of target cost.

### setAutoResetOptimizers()

Set if the optimizer resets at the end of iterations.

```
OneVsAll:setAutoResetOptimizers(option: boolean)
```

#### Parameters:

* option: A boolean value that specifies if optimizers resets at the end of iterations.

### setPrintOutput()

Set if the OneVsAll object prints output.

```
OneVsAll:setPrintOutput(option: boolean)
```

#### Parameters:

* option: A boolean value that specifies if the output is printed.

## Inherited From

* [BaseInstance](../Cores/BaseInstance.md)