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
| [KNearestNeighbours](Models/KNearestNeighbours.md)                                                               | KNN                    | Recommendation System                                                       |
| [LogisticRegression](Models/LogisticRegression.md)                                                               | None                   | Sales Prediction, Confidence Prediction                                     |
| [SupportVectorMachine](Models/SupportVectorMachine.md)                                                           | SVM                    | Hacking Detection, Anomaly Detection                                        |
| [NaiveBayes](Models/NaiveBayes.md)                                                                               | None                   | Text Classification                                                         |
| [NeuralNetwork](Models/NeuralNetwork.md)                                                                         | Multi-Layer Perceptron | Decision-Making, Player Behaviour Prediction                                |

### Non-Sequential + Reinforcement Learning

| Model                                                                                                                            | Alternate Names                           | Use Cases                                                                   |
|----------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------|-----------------------------------------------------------------------------|
| [QLearningNeuralNetwork](Models/QLearningNeuralNetwork.md)                                                                       | DQN, Deep Q-Learning                      | Self-Learning Fighting AIs, Self-Learning Parkouring AIs, Self-Driving Cars |
| [DoubleQLearningNeuralNetworkV1](Models/DoubleQLearningNeuralNetworkV1.md)                                                       | Double DQN, Double Deep Q-Learning (2010) | Same As Q-Learning Neural Network                                           |
| [DoubleQLearningNeuralNetworkV2](Models/DoubleQLearningNeuralNetworkV2.md)                                                       | Double DQN, Double Deep Q-Learning (2015) | Same As Q-Learning Neural Network                                           |
| [ClippedDoubleQLearningNeuralNetwork](Models/ClippedDoubleQLearningNeuralNetwork.md)                                             | Clipped Double Deep Q-Learning            | Same As Q-Learning Neural Network                                           |
| [StateActionRewardStateActionNeuralNetwork](Models/StateActionRewardStateActionNeuralNetwork.md)                                 | Deep SARSA                                | Same As Q-Learning Neural Network                                           |
| [DoubleStateActionRewardStateActionNeuralNetworkV1](Models/DoubleStateActionRewardStateActionNeuralNetworkV1.md)                 | Double Deep SARSA                         | Same As Q-Learning Neural Network                                           |
| [DoubleStateActionRewardStateActionNeuralNetworkV2](Models/DoubleStateActionRewardStateActionNeuralNetworkV2.md)                 | Double Deep SARSA                         | Same As Q-Learning Neural Network                                           |
| [ExpectedStateActionRewardStateActionNeuralNetwork](Models/ExpectedStateActionRewardStateActionNeuralNetwork.md)                 | Deep Expected SARSA                       | Same As Q-Learning Neural Network                                           |
| [DoubleExpectedStateActionRewardStateActionNeuralNetworkV1](Models/DoubleExpectedStateActionRewardStateActionNeuralNetworkV1.md) | Double Deep Expected SARSA                | Same As Q-Learning Neural Network                                           |
| [DoubleExpectedStateActionRewardStateActionNeuralNetworkV2](Models/DoubleExpectedStateActionRewardStateActionNeuralNetworkV2.md) | Double Deep Expected SARSA                | Same As Q-Learning Neural Network                                           |
| [ActorCritic](Models/ActorCritic.md)                                                                                             | AC                                        | Same As Q-Learning Neural Network                                           |
| [AdvantageActorCritic](Models/AdvantageActorCritic.md)                                                                           | A2C                                       | Same As Q-Learning Neural Network                                           |
| [AsynchronousAdvantageActorCritic](Models/AsynchronousAdvantageActorCritic.md)                                                   | A3C                                       | Same As Q-Learning Neural Network                                           |
| [REINFORCENeuralNetwork](Models/REINFORCENeuralNetwork.md)                                                                       | None                                      | Same As Q-Learning Neural Network                                           |
| [DuelingQLearning](Models/DuelingQLearning.md)                                                                                   | None                                      | Same As Q-Learning Neural Network                                           |
| [ProximalPolicyOptimization](Models/ProximalPolicyOptimization.md)                                                               | PPO                                       | Same As Q-Learning Neural Network                                           |
| [ProximalPolicyOptimizationClip](Models/ProximalPolicyOptimizationClip.md)                                                       | PPO-Clip                                  | Same As Q-Learning Neural Network                                           |
| [VanillaPolicyGradient](Models/VanillaPolicyGradient.md)                                                                         | VPG                                       | Same As Q-Learning Neural Network                                           |

### Sequential

| Model                                                                                                            | Alternate Names        | Use Cases                                                                   |
|------------------------------------------------------------------------------------------------------------------|------------------------|-----------------------------------------------------------------------------|
| [LongShortTermMemory](Models/LongShortTermMemory.md) (Deprecated)                                                | LSTM                   | Text Generation, Text Analysis                                              |
| [RecurrentNeuralNetwork](Models/RecurrentNeuralNetwork.md) (Deprecated)                                          | RNN                    | Same As Long Short Term Memory                                              |

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

## Generative

| Model                                                                                                                  | Alternate Names | Use Cases                             |
|------------------------------------------------------------------------------------------------------------------------|-----------------|---------------------------------------|
| [GenerativeAdversarialNetwork](Models/GenerativeAdversarialNetwork.md)                                                 | GAN             | Building And Art Generation           |

 

## BaseModels

[BaseModel](Models/BaseModel.md)

[GradientMethodBaseModel](Models/GradientMethodBaseModel.md)

[ReinforcementLearningNeuralNetworkBaseModel](Models/ReinforcementLearningNeuralNetworkBaseModel.md)

[ReinforcementLearningActorCriticBaseModel](Models/ReinforcementLearningActorCriticBaseModel.md)
