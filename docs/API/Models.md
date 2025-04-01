# [API Reference](../API.md) - Models

## Regression

| Model                                                                                       | Alternate Names | Use Cases                                     |
|---------------------------------------------------------------------------------------------|-----------------|-----------------------------------------------|
| [LinearRegression](Models/LinearRegression.md)                                              | None            | Price Prediction, Time To Level Up Prediction |
| [NormalLinearRegression](Models/NormalLinearRegression.md) (Not Recommended)                | None            | Same As Above                                 |
| [SupportVectorRegression](Models/SupportVectorRegression.md) (May Need Further Refinements) | SVR             | Same As Above                                 |

## Classification

| Model                                                                                                            | Alternate Names        | Use Cases                                                                   |
|------------------------------------------------------------------------------------------------------------------|------------------------|-----------------------------------------------------------------------------|
| [KNearestNeighbours](Models/KNearestNeighbours.md)                                                               | KNN                    | Recommendation System                                                       |
| [LogisticRegression](Models/LogisticRegression.md)                                                               | Perceptron             | Sales Prediction, Confidence Prediction                                     |
| [SupportVectorMachine](Models/SupportVectorMachine.md)                                                           | SVM                    | Hacking Detection, Anomaly Detection                                        |
| [GaussianNaiveBayes](Models/GaussianNaiveBayes.md)                                                               | None                   | Text Classification                                                         |
| [MultinomialNaiveBayes](Models/MultinomialNaiveBayes.md)                                                         | None                   | Text Classification                                                         |
| [BernoulliNaiveBayes](Models/BernoulliNaiveBayes.md)                                                             | None                   | Text Classification                                                         |
| [ComplementNaiveBayes](Models/ComplementNaiveBayes.md)                                                           | None                   | Text Classification With Imbalanced Classes Count                           |
| [NeuralNetwork](Models/NeuralNetwork.md)                                                                         | Multi-Layer Perceptron | Decision-Making, Player Behaviour Prediction                                |

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

## Deep Reinforcement Learning

| Model                                                                                                                            | Alternate Names                           | Use Cases                                                                   |
|----------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------|-----------------------------------------------------------------------------|
| [DeepQLearning](Models/DeepQLearning.md)                                                                                         | Deep Q Network                            | Self-Learning Fighting AIs, Self-Learning Parkouring AIs, Self-Driving Cars |
| [DeepDoubleQLearningV1](Models/DeepDoubleQLearningV1.md)                                                                         | Double Deep Q Network (2010)              | Same As Deep Q-Learning                                                     |
| [DeepDoubleQLearningV2](Models/DeepDoubleQLearningV2.md)                                                                         | Double Deep Q Network (2015)              | Same As Deep Q-Learning                                                     |
| [DeepClippedDoubleQLearning](Models/DeepClippedDoubleQLearning.md)                                                               | Clipped Double Deep Q Network             | Same As Deep Q-Learning                                                     |
| [DeepStateActionRewardStateAction](Models/DeepStateActionRewardStateAction.md)                                                   | Deep SARSA                                | Same As Deep Q-Learning                                                     |
| [DeepDoubleStateActionRewardStateActionV1](Models/DeepDoubleStateActionRewardStateActionV1.md)                                   | Double Deep SARSA                         | Same As Deep Q-Learning                                                     |
| [DeepDoubleStateActionRewardStateActionV2](Models/DeepDoubleStateActionRewardStateActionV2.md)                                   | Double Deep SARSA                         | Same As Deep Q-Learning                                                     |
| [DeepExpectedStateActionRewardStateAction](Models/DeepExpectedStateActionRewardStateAction.md)                                   | Deep Expected SARSA                       | Same As Deep Q-Learning                                                     |
| [DeepDoubleExpectedStateActionRewardStateActionV1](Models/DeepDoubleExpectedStateActionRewardStateActionV1.md)                   | Double Deep Expected SARSA                | Same As Deep Q-Learning                                                     |
| [DeepDoubleExpectedStateActionRewardStateActionV2](Models/DeepDoubleExpectedStateActionRewardStateActionV2.md)                   | Double Deep Expected SARSA                | Same As Deep Q-Learning                                                     |
| [ActorCritic](Models/ActorCritic.md)                                                                                             | AC                                        | Same As Deep Q-Learning                                                     |
| [AdvantageActorCritic](Models/AdvantageActorCritic.md)                                                                           | A2C                                       | Same As Deep Q-Learning                                                     |
| [REINFORCE](Models/REINFORCE.md)                                                                                                 | None                                      | Same As Deep Q-Learning                                                     |
| [MonteCarloControl](Models/MonteCarloControl.md)                                                                                 | None                                      | Same As Deep Q-Learning                                                     |
| [OffPolicyMonteCarloControl](Models/OffPolicyMonteCarloControl.md)                                                               | None                                      | Same As Deep Q-Learning                                                     |
| [VanillaPolicyGradient](Models/VanillaPolicyGradient.md)                                                                         | VPG                                       | Same As Deep Q-Learning                                                     |
| [ProximalPolicyOptimization](Models/ProximalPolicyOptimization.md)                                                               | PPO                                       | Same As Deep Q-Learning                                                     |
| [ProximalPolicyOptimizationClip](Models/ProximalPolicyOptimizationClip.md)                                                       | PPO-Clip                                  | Same As Deep Q-Learning                                                     |
| [SoftActorCritic](Models/SoftActorCritic.md)                                                                                     | SAC                                       | Same As Deep Q-Learning                                                     |
| [DeepDeterministicPolicyGradient](Models/DeepDeterministicPolicyGradient.md)                                                     | DDPG                                      | Same As Deep Q-Learning                                                     |
| [TwinDelayedDeepDeterministicPolicyGradient](Models/TwinDelayedDeepDeterministicPolicyGradient.md)                               | TD3                                       | Same As Deep Q-Learning                                                     |

## Generative

| Model                                                                                                                  | Alternate Names | Use Cases                             |
|------------------------------------------------------------------------------------------------------------------------|-----------------|---------------------------------------|
| [GenerativeAdversarialNetwork](Models/GenerativeAdversarialNetwork.md)                                                 | GAN             | Building And Image Generation         |
| [ConditionalGenerativeAdversarialNetwork](Models/ConditionalGenerativeAdversarialNetwork.md)                           | CGAN            | Same As GAN, But Can Assign Classes   |
| [WassersteinGenerativeAdversarialNetwork](Models/WassersteinGenerativeAdversarialNetwork.md)                           | WGAN            | Same As GAN, But More Stable          |
| [ConditionalWassersteinGenerativeAdversarialNetwork](Models/ConditionalWassersteinGenerativeAdversarialNetwork.md)     | CWGAN           | Combination Of Both CGAN And WGAN     |

## BaseModels

[BaseModel](Models/BaseModel.md)

[GradientMethodBaseModel](Models/GradientMethodBaseModel.md)

[IterativeMethodBaseModel](Models/IterativeMethodBaseModel.md)

[ReinforcementLearningBaseModel](Models/ReinforcementLearningBaseModel.md)

[ReinforcementLearningActorCriticBaseModel](Models/ReinforcementLearningActorCriticBaseModel.md)
