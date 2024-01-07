# [API Reference](../../API.md) - [Others](../Others.md) - ModelDatasetCreator

Creates a dataset for models to train on

## Constructors

### new()

Creates a new gradient descent modifier object. If any of the arguments are not given, default argument values for that argument will be used.

```
ModelDatasetCreator.new(): ModelDatasetCreatorObject
```

## Functions

### setDatasetSplitPercentages()

Set the split percentages for training, validation and testing. The current default values are 0.7 for training and 0.3 for testing.

```
ModelDatasetCreator:setDatasetSplitPercentages(trainDataPercentage: number, validationDataPercentage: number, testDataPercentage: number)
```

#### Parameters:

* trainDataPercentage: The percentage of dataset to be turned to training data. The value must be between 0 and 1.

* validationDataPercentage: The percentage of dataset to be turned to validation data. The value must be between 0 and 1.

* testDataPercentage: The percentage of dataset to be turned to testing data. The value must be between 0 and 1.

### setDatasetRandomizationProperties()

Set the split percentages for training, validation and testing. The current default values are 0.7 for training and 0.3 for testing.

```
ModelDatasetCreator:setDatasetRandomizationProperties(randomizationProbabilityThreshold: number)
```

#### Parameters:

* randomizationProbabilityThreshold: The probability to randomize the positions of each datapoints. The value must be between 0 and 1.

### randomizeDataset()

Randomizes the each data positions in te dataset 

```
ModelDatasetCreator:randomizeDataset(featureMatrix: matrix, labelVectorOrMatrix: matrix): matrix, matrix
```

#### Parameters:

* featureMatrix: The matrix containing all the data.

* labelVectorOrMatrix: The matrix containing all the label values related to feature matrix. Optional argument.

#### Returns:

* randomizedFeatureMatrix: The matrix containing all the data.

* randomizedLabelVectorOrMatrix: The matrix containing all the label values related to feature matrix. Only returns if labelVectorOrMatrix is added.

### splitDataset()

Predict the values for given data.

```
ModelDatasetCreator:splitDataset(datasetMatrix): matrix, matrix, matrix
```

#### Parameters:

* datasetMatrix: The feature matrix or label matrix to split.

#### Returns:

* trainDatasetMatrix: The dataset for training models.

* validationDatasetMatrix: The dataset for validating models.

* testDatasetMatrix: The dataset for testing models.
