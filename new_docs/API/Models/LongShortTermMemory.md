# [API Reference](../../API.md) - [Models](../Models.md) - LongShortTermMemory (LSTM)

LongShortTermMemory is a supervised machine learning model that predicts a sequence of positive numbers of discrete values (a.k.a. tokens).

## Stored Model Parameters

Contains a table of matrices.  

* ModelParameters[1] = Forget gate weight matrix

* ModelParameters[2] = Forget gate bias matrix

* ModelParameters[3] = Save gate weight matrix

* ModelParameters[4] = Save gate bias matrix

* ModelParameters[5] = Tanh weight matrix
	
* ModelParameters[6] = Tanh bias matrix
	
* ModelParameters[7] = Focus gate weight matrix
	
* ModelParameters[8] = Focus gate bias matrix
	
* ModelParameters[9] = Output weight matrix
	
* ModelParameters[10] = Output bias matrix

## Constructors

### new()

Create new model object. If any of the arguments are nil, default argument values for that argument will be used.

```
LongShortTermMemory.new(maxNumberOfIterations: integer, learningRate: number, targetCost: number): ModelObject
```

#### Parameters:

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* targetCost: The cost at which the model stops training.

#### Returns:

* ModelObject: The generated model object.

## Functions

### setParameters()

Set model's parameters. When any of the arguments are nil, previous argument values for that argument will be used.

```
LongShortTermMemory:setParameters(maxNumberOfIterations: integer, learningRate: number, targetCost: number)
```

#### Parameters:

* tokenSize: The number of unique tokens that will be inputed or generated by the model.

* maxNumberOfIterations: How many times should the model needed to be trained.

* learningRate: The speed at which the model learns. Recommended that the value is set between (0 to 1).

* targetCost: The cost at which the model stops training.

### createLayers()

Create layers of the recurrent neural network.

```
LongShortTermMemory:createLayers(inputSize: integer, hiddenSize: integer, outputSize: integer)
```

* inputSize: The number of unique tokens to be inputted to the model.

* hiddenSize: The number of neurons inside the model.

* outputSize: The number of unique tokens to be generated by the model.

### setOptimizers()

Add optimizers to different matrices.

```
LongShortTermMemory:setOptimizers(ForgetGateWeightOptimizer: OptimizerObject, SaveGateWeightOptimizer: OptimizerObject, TanhWeightOptimizer: OptimizerObject, FocusGateOptimizer: OptimizerObject, OutputWeightOptimizer: OptimizerObject, ForgetGateBiasOptimizer: OptimizerObject, SaveGateBiasOptimizer: OptimizerObject, TanhBiasOptimizer: OptimizerObject, FocusGateBiasOptimizer: OptimizerObject, OutputBiasOptimizer: OptimizerObject)
```

#### Parameters:

* ForgetGateWeightOptimizer: Optimizer object for forget gate calculations.

* SaveGateWeightOptimizer: Optimizer object for save gate calculations.
  
* TanhWeightOptimizer: Optimizer object for tanh calculations.

* FocusGateOptimize: Optimizer object for focus gate calculations.
  
* OutputWeightOptimizer: Optimizer object for output layer calculations.

* ForgetGateBiasOptimizer: Optimizer object for forget gate bias calculations.

* SaveGateBiasOptimizer: Optimizer object for save gate bias calculations.

* TanhBiasOptimizer: Optimizer object for tanh bias calculations.

* FocusGateBiasOptimizer: Optimizer object for focus gate bias calculations.

* OutputBiasOptimizer: Optimizer object for output bias layer calculations.

### setRegularizations()

Add regularizations to different matrices.

```
LongShortTermMemory:setRegularizations(ForgetGateWeightRegularization: RegularizationObject, SaveGateWeightRegularization: RegularizationObject, TanhWeightRegularization: RegularizationObject, FocusGateRegularization: RegularizationObject, OutputWeightRegularization: RegularizationObject, ForgetGateBiasRegularization: RegularizationObject, SaveGateBiasRegularization: RegularizationObject, TanhBiasRegularization: RegularizationObject, FocusGateBiasRegularization: RegularizationObject, OutputBiasRegularization: RegularizationObject)
```

#### Parameters:

* ForgetGateWeightRegularization: Regularization object for forget gate calculations.

* SaveGateWeightRegularization: Regularization object for save gate calculations.
  
* TanhWeightRegularization: Regularization object for tanh calculations.

* FocusGateOptimize: Regularization object for focus gate calculations.
  
* OutputWeightRegularization: Regularization object for output layer calculations.

* ForgetGateBiasRegularization: Regularization object for forget gate bias calculations.

* SaveGateBiasRegularization: Regularization object for save gate bias calculations.

* TanhBiasRegularization: Regularization object for tanh bias calculations.

* FocusGateBiasRegularization: Regularization object for focus gate bias calculations.

* OutputBiasRegularization: Regularization object for output bias layer calculations.

### train()

Train the model. 

```
LongShortTermMemory:train(tableOfTokenInputSequenceArray: tokenInputSequenceArray[], tableOfTokenOutputSequenceArray: tokenOutputSequenceArray[]): number[]
```
#### Parameters:

* tableOfTokenInputSequenceArray: An array containing sequences of tokens.

* tableOfTokenOutputSequenceArray: An array containing sequences of tokens. Leave this empty if you want to use tokenInputSequenceArray only.

#### Returns:

* costArray: An array containing cost values.

### predict()

Predict a sequence of output tokens for a given sequence of input tokens.

```
LongShortTermMemory:predict(tableOfTokenInputSequenceArray: tokenInputSequenceArray[], returnOriginalOutput: boolean): tokenOutputSequenceArray[]
```

#### Parameters:

* tableOfTokenInputSequenceArray: An array containing sequences of tokens.

* returnOriginalOutput: Set whether or not to return predicted matrices instead of value with highest probability.

#### Returns:

* tableOfTokenOutputSequenceArray: An array containing sequences of tokens.

## Notes:

* Ensure that the length of input tokens is equal to ouput tokens if output tokens are used.

* For an uneven lengths of tokens, I recommend that you "pad" the token arrays with 0. For example, if we have input tokens {1, 3, 4} and output tokens {6, 2, 3, 4}, then use {1, 3, 4, 0, 0, 0, 0} and {0, 0, 0, 6, 2, 3, 4}.

## Inherited From

* [BaseModel](BaseModel.md)
