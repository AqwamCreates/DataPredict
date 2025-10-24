# [API Reference](../../API.md) - [Others](../Others.md) - DatasetCreator

Modifies existing dataset so that it can be used by the models.

## Constructors

### new()

Creates a ModelDatasetCreator object.

```
DatasetCreator.new(): ModelDatasetCreatorObject
```

## Functions

### setDatasetSplitPercentages()

Set the split percentages for training, validation and testing. The current default values are 0.7 for training and 0.3 for testing.

```
DatasetCreator:setDatasetSplitPercentages(trainDataPercentage: number, validationDataPercentage: number, testDataPercentage: number)
```

#### Parameters:

* trainDataRatio: The ratio of dataset to be turned to training data. The value must be between 0 and 1.

* validationDataRatio: The ratio of dataset to be turned to validation data. The value must be between 0 and 1.

* testDataRatio: The ratio of dataset to be turned to testing data. The value must be between 0 and 1.

### setDatasetRandomizationProbability()

```
DatasetCreator:setDatasetRandomizationProbability(datasetRandomizationProbability: number)
```

#### Parameters:

* datasetRandomizationProbability: The probability to randomize the positions of each datapoints. The higher the value, the more likely it will be randomized. The value must be between 0 and 1. 

### randomizeDataset()

Randomizes the each data positions in te dataset 

```
DatasetCreator:randomizeDataset(featureMatrix: matrix, labelVectorOrMatrix: matrix): matrix, matrix
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
DatasetCreator:splitDataset(datasetMatrix): matrix, matrix, matrix
```

#### Parameters:

* datasetMatrix: The feature matrix or label matrix to split.

#### Returns:

* trainDatasetMatrix: The dataset for training models.

* validationDatasetMatrix: The dataset for validating models.

* testDatasetMatrix: The dataset for testing models.

## Inherited From

* [BaseInstance](../Cores/BaseInstance.md)
