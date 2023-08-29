# [API Reference](../API.md) - Models

## Regression

| Model                                                                        | Alternate Names | Use Cases                                     |
|------------------------------------------------------------------------------|-----------------|-----------------------------------------------|
| [LinearRegression](Models/LinearRegression.md)                               | None            | Price Prediction, Time To Level Up Prediction |
| [NormalLinearRegression](Models/NormalLinearRegression.md) (Not Recommended) | None            | Same As Above                                 |

## Classification

| Model                                                                                                            | Alternate Names        | Use Cases                                                                   |
|------------------------------------------------------------------------------------------------------------------|------------------------|-----------------------------------------------------------------------------|
| [LogisticRegression](Models/LogisticRegression.md)                                                               | None                   | Sales Prediction                                                            |
| [SupportVectorMachine](Models/SupportVectorMachine.md)                                                           | None                   | Hacking Detection, Anomaly Detection                                        |
| [NaiveBayes](Models/NaiveBayes.md)                                                                               | None                   | Text Classification                                                         |
| [NeuralNetwork](Models/NeuralNetwork.md)                                                                         | Multi-Layer Perceptron | Decision-Making, Fake Players                                               |
| [QLearningNeuralNetwork](Models/QLearningNeuralNetwork.md)                                                       | DQN, Deep Q-Learning   | Self-Learning Fighting AIs, Self-Learning Parkouring AIs, Self-Driving Cars |
| [StateActionRewardStateActionNeuralNetwork](Models/StateActionRewardStateActionNeuralNetwork.md)                 | Deep SARSA             | Same As Q-Learning Neural Network                                           |
| [ExpectedStateActionRewardStateActionNeuralNetwork](Models/ExpectedStateActionRewardStateActionNeuralNetwork.md) | Deep Expected SARSA    | Same As Q-Learning Neural Network                                           |
| [ReinforcingNeuralNetwork](Models/ReinforcingNeuralNetwork.md)                                                   | None                   | Same As Q-Learning Neural Network                                           |
| [LongShortTermMemory](Models/LongShortTermMemory.md)                                                             | LSTM                   | Text Generation, Text Analysis                                              |
| [RecurrentNeuralNetwork](Models/RecurrentNeuralNetwork.md)                                                       | RNN                    | Same As Long Short Term Memory                                              |




## Clustering

[AffinityPropagation](Models/AffinityPropagation.md)

[AgglomerativeHierarchical](Models/AgglomerativeHierarchical.md)

[DensityBasedSpatialClusteringOfApplicationsWithNoise](Models/DensityBasedSpatialClusteringOfApplicationsWithNoise.md) - a.k.a. DBSCAN

[MeanShift](Models/MeanShift.md)

[ExpectationMaximization](Models/ExpectationMaximization.md) - a.k.a. EM

[KMeans](Models/KMeans.md)

[KMedoids](Models/KMedoids.md)

## BaseModels

[BaseModel](Models/BaseModel.md)
