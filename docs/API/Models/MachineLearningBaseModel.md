# API Reference - Models - MachineLearningBaseModel

The base model for all machine and deep learning models.

## Constructors

### new()

Creates a new machine learning base model

```
MachineLearningBaseModel.new(): BaseModelObject
```

## Functions

### getModelParameters()

Gets the model parameters from the base model.

```
MachineLearningBaseModel:getModelParameters(): ModelParameters
```

#### Returns

* ModelParameters: The model parameters fetched from the base model

### setModelParameters()

Set the model parameters to the base model.

```
MachineLearningBaseModel:setModelParameters(ModelParameters: ModelParametersMatrix)
```

#### Parameters
* ModelParameters: A matrix containing model parameters to be given to the base model.

### clearModelParameters()

Clears the model parameters contained inside base model.

```
MachineLearningBaseModel:clearModelParameters()
```

### clearLastPredictedOutput()

Clears the last predicted outputs made by the base model.

```
MachineLearningBaseModel:clearLastPredictedOutput()
```

### clearLastCalculations()

Clears last calculation made by the base model.

```
MachineLearningBaseModel:clearLastCalculations()
```

### clearLastPredictedOutputAndCalculations()

Clears the last predicted outputs and calculation made by the base model.

```
MachineLearningBaseModel:clearLastPredictedOutputAndCalculations()
```

### setPrintOutput()

Set if the model prints output.

```
MachineLearningBaseModel:setPrintOutput(option: boolean)
```

#### Parameters:

* option: A boolean value that specifies if the output is printed


