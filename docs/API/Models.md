# [API Reference](../API.md) - Models

## Regression

| Model                                                                        | Alternate Names | Use Cases                                     |
|------------------------------------------------------------------------------|-----------------|-----------------------------------------------|
| [LinearRegression](Models/LinearRegression.md)                               | None            | Price Prediction, Time To Level Up Prediction |
| [NormalLinearRegression](Models/NormalLinearRegression.md) (Not Recommended) | None            | Same As Above                                 |

## Classification

### Non-Sequential

| Model                                                                                                            | Alternate Names        | Use Cases                                                                   |
|------------------------------------------------------------------------------------------------------------------|------------------------|-----------------------------------------------------------------------------|
| [LogisticRegression](Models/LogisticRegression.md)                                                               | None                   | Sales Prediction, Confidence Prediction                                     |
| [SupportVectorMachine](Models/SupportVectorMachine.md)                                                           | SVM                    | Hacking Detection, Anomaly Detection                                        |
| [NaiveBayes](Models/NaiveBayes.md)                                                                               | None                   | Text Classification                                                         |
| [NeuralNetwork](Models/NeuralNetwork.md)                                                                         | Multi-Layer Perceptron | Decision-Making, Player Behaviour Prediction                                |

### Non-Sequential + Reinforcement Learning

| Model                                                                                                            | Alternate Names                      | Use Cases                                                                   |
|------------------------------------------------------------------------------------------------------------------|--------------------------------------|-----------------------------------------------------------------------------|
| [QLearningNeuralNetwork](Models/QLearningNeuralNetwork.md)                                                       | DQN, Deep Q-Learning                 | Self-Learning Fighting AIs, Self-Learning Parkouring AIs, Self-Driving Cars |
| [DoubleQLearningNeuralNetworkV1](Models/DoubleQLearningNeuralNetworkV1.md)                                       | DDQN, Double Deep Q-Learning (2010)  | Same As Q-Learning Neural Network                                           |
| [DoubleQLearningNeuralNetworkV2](Models/DoubleQLearningNeuralNetworkV2.md)                                       | DDQN, Double Deep Q-Learning (2015)  | Same As Q-Learning Neural Network                                           |
| [ClippedDoubleQLearningNeuralNetwork](Models/ClippedDoubleQLearningNeuralNetwork.md)                             | Clipped Double Deep Q-Learning       | Same As Q-Learning Neural Network                                           |
| [StateActionRewardStateActionNeuralNetwork](Models/StateActionRewardStateActionNeuralNetwork.md)                 | Deep SARSA                           | Same As Q-Learning Neural Network                                           |
| [ExpectedStateActionRewardStateActionNeuralNetwork](Models/ExpectedStateActionRewardStateActionNeuralNetwork.md) | Deep Expected SARSA                  | Same As Q-Learning Neural Network                                           |
| [ReinforcingNeuralNetwork](Models/ReinforcingNeuralNetwork.md)                                                   | None                                 | Same As Q-Learning Neural Network                                           |

### Sequential

| Model                                                                                                            | Alternate Names        | Use Cases                                                                   |
|------------------------------------------------------------------------------------------------------------------|------------------------|-----------------------------------------------------------------------------|
| [LongShortTermMemory](Models/LongShortTermMemory.md)                                                             | LSTM                   | Text Generation, Text Analysis                                              |
| [RecurrentNeuralNetwork](Models/RecurrentNeuralNetwork.md)                                                       | RNN                    | Same As Long Short Term Memory                                              |

## Clustering

| Model                                                                                                                  | Alternate Names | Use Cases                             |
|------------------------------------------------------------------------------------------------------------------------|-----------------|---------------------------------------|
| [AffinityPropagation](Models/AffinityPropagation.md)                                                                   | None            | Player Grouping                       |
| [AgglomerativeHierarchical](Models/AgglomerativeHierarchical.md)                                                       | None            | Similarity Grouping                   |
| [DensityBasedSpatialClusteringOfApplicationsWithNoise](Models/DensityBasedSpatialClusteringOfApplicationsWithNoise.md) | DBSCAN          | Density Grouping                      |
| [MeanShift](Models/MeanShift.md)                                                                                       | None            | Center Of Data Search                 |
| [ExpectationMaximization](Models/ExpectationMaximization.md)                                                           | EM              | Hacking Detection, Anomaly Detection  |
| [KMeans](Models/KMeans.md)                                                                                             | None            | Market Segmentation, General Grouping |
| [KMedoids](Models/KMedoids.md)                                                                                         | None            | Same as K-Means                       |

## BaseModels

[BaseModel](Models/BaseModel.md)

[ReinforcementLearningNeuralNetworkBaseModel](Models/ReinforcementLearningNeuralNetworkBaseModel.md)
