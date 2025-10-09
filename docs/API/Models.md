# [API Reference](../API.md) - Models

If you wonder what are the most high-value use cases that helps with retention and revenue generation with this DataPredict™, you can view them [here](../HighValueProjectTutorials.md)!

| Model Type                     | Count |
|--------------------------------|-------|
| Regression                     | 5     |
| Classification                 | 13    |
| Clustering                     | 8     |
| Deep Reinforcement Learning    | 21    |
| Tabular Reinforcement Learning | 5     |
| Generative                     | 4     |
| Total                          | 57    |

For strong deep learning applications, have a look at [DataPredict™ Neural](https://aqwamcreates.github.io/DataPredict-Neural) (object-oriented) and [DataPredict™ Axon](https://aqwamcreates.github.io/DataPredict-Axon) (function-oriented) instead.

  * Contains all the deep reinforcement learning and generative algorithms listed here.

  * Includes convolutional, pooling, dropout and activation layers.

### Note

Currently, these algorithms lack online learning capabilities. We're still looking into it.

| Model Type     | Model Names                                                                         |
|----------------|-------------------------------------------------------------------------------------|
| Regression     | NormalLinearRegression, SupportVectorRegression                                     |
| Classification | SupportVectorMachine, OneClassSupportVectorMachine                                  |
| Clustering     | KMedoids, AffinityPropagation, DensityBasedSpatialClusteringOfApplicationsWithNoise |

This means DataPredict™ has ~90% (50 out of 57) models with online learning capabilities.

By default, most models would perform offline / batch training on the first train, but then switches to online / sequential / incremental after the first train.

## Regression

| Model                                                                                      | Alternate Names | Use Cases                                                                       |
|--------------------------------------------------------------------------------------------|-----------------|---------------------------------------------------------------------------------|
| [LinearRegression](Models/LinearRegression.md) (Beginner Algorithm)                        | None            | General Time-To-Leave Prediction And In-Game Currency Price Generation          |
| [PassiveAggressiveRegressor](Models/PassiveAggressiveRegressor.md)                         | PA-R            | Fast Constrained Time-To-Leave Prediction And In-Game Currency Price Generation |
| [SupportVectorRegression](Models/SupportVectorRegression.md)                               | SVR             | Constrained Time-To-Leave Prediction And In-Game Currency Price Generation      |
| [KNearestNeighboursRegressor](Models/KNearestNeighboursRegressor.md)                       | KNN-R           | Memory-Based Time-To-Leave Prediction And In-Game Currency Price Generation     |
| [NormalLinearRegression](Models/NormalLinearRegression.md) (Not Recommended)               | None            | Final Solution Time-To-Leave Prediction And In-Game Currency Price Generation   |

## Classification

| Model                                                                                                            | Alternate Names        | Use Cases                                                                                |
|------------------------------------------------------------------------------------------------------------------|------------------------|------------------------------------------------------------------------------------------|
| [LogisticRegression](Models/LogisticRegression.md) (Beginner Algorithm)                                          | Perceptron             | Probability-To-Leave Prediction, Player Churn Prediction, Confidence Prediction          |
| [PassiveAggressiveClassifier](Models/PassiveAggressiveClassifier.md)                                             | PA-C                   | Fast Purchase Likelihood Estimation, Decision Making                                     |
| [OneClassPassiveAggressiveClassifier](Models/OneClassPassiveAggressiveClassifier.md)                             | OC-PA-C                | Fast Hacking Detection, Anomaly Detection (Using Single Class Data)                      |
| [NearestCentroid](Models/NearestCentroid.md)                                                                     | NC                     | Fast Grouping Or Quick Decision Making                                                   |
| [KNearestNeighboursClassifier](Models/KNearestNeighboursClassifier.md)                                           | KNN-C                  | Item Recommendation, Similar Player Matchmaking                                          |
| [SupportVectorMachine](Models/SupportVectorMachine.md)                                                           | SVM                    | Hacking Detection, Anomaly Detection                                                     |
| [OneClassSupportVectorMachine](Models/OneClassSupportVectorMachine.md)                                           | OC-SVM                 | Hacking Detection, Anomaly Detection (Using Single Class Data)                           |
| [NeuralNetwork](Models/NeuralNetwork.md) (Beginner Algorithm)                                                    | Multi-Layer Perceptron | Decision-Making, Player Behaviour Prediction                                             |
| [GaussianNaiveBayes](Models/GaussianNaiveBayes.md) (Stonger As Generative Model)                                 | None                   | Player Behavior Categorization (e.g. Cautious Vs. Aggressive), Fast State Classification |
| [MultinomialNaiveBayes](Models/MultinomialNaiveBayes.md) (Stonger As Generative Model)                           | None                   | Inventory Action Prediction, Strategy Profiling Based on Item Usage                      |
| [BernoulliNaiveBayes](Models/BernoulliNaiveBayes.md) (Stonger As Generative Model)                               | None                   | Binary Action Prediction (e.g. Jump Or Not), Quick Decision Filters                      |
| [ComplementNaiveBayes](Models/ComplementNaiveBayes.md) (Stonger As Generative Model)                             | None                   | Imbalanced Class Prediction (e.g. Rare Choices, Niche Paths)                             |
| [CategoricalNaiveBayes](Models/CategoricalNaiveBayes.md) (Stonger As Generative Model)                           | None                   | Player Choice Prediction (e.g. Weapon Type, Character Class, Map Region Selection)       |

## Clustering

| Model                                                                                                                  | Alternate Names | Use Cases                                                            |
|------------------------------------------------------------------------------------------------------------------------|-----------------|----------------------------------------------------------------------|
| [KMeans](Models/KMeans.md) (Beginner Algorithm)                                                                        | None            | Maximizing Area-of-Effect Abilities, Target Grouping                 |
| [FuzzyCMeans](Models/FuzzyCMeans.md)                                                                                   | None            | Overlapping Area-of-Effect Abilities, Overlapping Target Grouping    |
| [KMedoids](Models/KMedoids.md)                                                                                         | None            | Player Grouping Based On Player Locations With Leader Identification |
| [AgglomerativeHierarchical](Models/AgglomerativeHierarchical.md)                                                       | None            | Enemy Difficulty Generation                                          |
| [ExpectationMaximization](Models/ExpectationMaximization.md)                                                           | EM              | Hacking Detection, Anomaly Detection                                 |
| [MeanShift](Models/MeanShift.md)                                                                                       | None            | Boss Spawn Location Search Based On Player Locations                 |
| [AffinityPropagation](Models/AffinityPropagation.md)                                                                   | None            | Player Grouping                                                      |
| [DensityBasedSpatialClusteringOfApplicationsWithNoise](Models/DensityBasedSpatialClusteringOfApplicationsWithNoise.md) | DBSCAN          | Density Grouping                                                     |

## Deep Reinforcement Learning

| Model                                                                                                                            | Alternate Names                           | Use Cases                                                                   |
|----------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------|-----------------------------------------------------------------------------|
| [DeepQLearning](Models/DeepQLearning.md)                                                                                         | Deep Q Network                            | Best Self-Learning Player AIs, Best Recommendation Systems                  |
| [DeepDoubleQLearningV1](Models/DeepDoubleQLearningV1.md)                                                                         | Double Deep Q Network (2010)              | Best Self-Learning Player AIs, Best Recommendation Systems                  |
| [DeepDoubleQLearningV2](Models/DeepDoubleQLearningV2.md)                                                                         | Double Deep Q Network (2015)              | Best Self-Learning Player AIs, Best Recommendation Systems                  |
| [DeepClippedDoubleQLearning](Models/DeepClippedDoubleQLearning.md)                                                               | Clipped Deep Double Q Network             | Best Self-Learning Player AIs, Best Recommendation Systems                  |
| [DeepStateActionRewardStateAction](Models/DeepStateActionRewardStateAction.md)                                                   | Deep SARSA                                | Safe Self-Learning Player AIs, Safe Recommendation Systems                  |
| [DeepDoubleStateActionRewardStateActionV1](Models/DeepDoubleStateActionRewardStateActionV1.md)                                   | Double Deep SARSA                         | Safe Self-Learning Player AIs, Safe Recommendation Systems                  |
| [DeepDoubleStateActionRewardStateActionV2](Models/DeepDoubleStateActionRewardStateActionV2.md)                                   | Double Deep SARSA                         | Safe Self-Learning Player AIs, Safe Recommendation Systems                  |
| [DeepExpectedStateActionRewardStateAction](Models/DeepExpectedStateActionRewardStateAction.md)                                   | Deep Expected SARSA                       | Balanced Self-Learning Player AIs, Balanced Recommendation Systems          |
| [DeepDoubleExpectedStateActionRewardStateActionV1](Models/DeepDoubleExpectedStateActionRewardStateActionV1.md)                   | Double Deep Expected SARSA                | Balanced Self-Learning Player AIs, Balanced Recommendation Systems          |
| [DeepDoubleExpectedStateActionRewardStateActionV2](Models/DeepDoubleExpectedStateActionRewardStateActionV2.md)                   | Double Deep Expected SARSA                | Balanced Self-Learning Player AIs, Balanced Recommendation Systems          |
| [DeepMonteCarloControl](Models/DeepMonteCarloControl.md) (May Need Further Refinement)                                           | None                                      | Online Self-Learning Player AIs                                             |
| [DeepOffPolicyMonteCarloControl](Models/DeepOffPolicyMonteCarloControl.md)                                                       | None                                      | Offline Self-Learning Player AIs                                            |
| [REINFORCE](Models/REINFORCE.md)                                                                                                 | None                                      | Reward-Based Self-Learning Player AIs                                       |
| [VanillaPolicyGradient](Models/VanillaPolicyGradient.md) (May Need Further Refinement)                                           | VPG                                       | Baseline-Based Self-Learning Player AIs                                     |
| [ActorCritic](Models/ActorCritic.md)                                                                                             | AC                                        | Critic-Based Self-Learning Player AIs                                       |
| [AdvantageActorCritic](Models/AdvantageActorCritic.md)                                                                           | A2C                                       | Advantage-Based Self-Learning Player AIs                                    |
| [ProximalPolicyOptimization](Models/ProximalPolicyOptimization.md)                                                               | PPO                                       | Industry-Grade And Research-Grade Self-Learning Player And Vehicle AIs      |
| [ProximalPolicyOptimizationClip](Models/ProximalPolicyOptimizationClip.md)                                                       | PPO-Clip                                  | Industry-Grade And Research-Grade Self-Learning Player And Vehicle AIs      |
| [SoftActorCritic](Models/SoftActorCritic.md)                                                                                     | SAC                                       | Self-Learning Vehicle AIs                                                   |
| [DeepDeterministicPolicyGradient](Models/DeepDeterministicPolicyGradient.md)                                                     | DDPG                                      | Self-Learning Vehicle AIs                                                   |
| [TwinDelayedDeepDeterministicPolicyGradient](Models/TwinDelayedDeepDeterministicPolicyGradient.md)                               | TD3                                       | Self-Learning Vehicle AIs                                                   |

## Tabular Reinforcement Learning

| Model                                                                                                        | Alternate Names | Use Cases                       |
|--------------------------------------------------------------------------------------------------------------|-----------------|---------------------------------|
| [TabularQLearning](Models/TabularQLearning.md)                                                               | Q-Learning      | Best Self-Learning Grid AIs     |
| [TabularStateActionRewardState](Models/TabularStateActionRewardStateAction.md) (May Need Further Refinement) | SARSA           | Safe Self-Learning Grid AIs     |
| [TabularExpectedStateActionRewardState](Models/TabularExpectedStateActionRewardStateAction.md)               | Expected SARSA  | Balanced Self-Learning Grid AIs |
| [TabularMonteCarloControl](Models/TabularMonteCarloControl.md) (May Need Further Refinement)                 | MC              | Online Self-Learning Grid AIs   |
| [TabularOffPolicyMonteCarloControl](Models/TabularOffPolicyMonteCarloControl.md)                             | Off-Policy MC   | Offline Self-Learning Grid AIs  |

## Generative

| Model                                                                                                                  | Alternate Names | Use Cases                             |
|------------------------------------------------------------------------------------------------------------------------|-----------------|---------------------------------------|
| [GenerativeAdversarialNetwork](Models/GenerativeAdversarialNetwork.md)                                                 | GAN             | Enemy Data Generation                 |
| [ConditionalGenerativeAdversarialNetwork](Models/ConditionalGenerativeAdversarialNetwork.md)                           | CGAN            | Same As GAN, But Can Assign Classes   |
| [WassersteinGenerativeAdversarialNetwork](Models/WassersteinGenerativeAdversarialNetwork.md)                           | WGAN            | Same As GAN, But More Stable          |
| [ConditionalWassersteinGenerativeAdversarialNetwork](Models/ConditionalWassersteinGenerativeAdversarialNetwork.md)     | CWGAN           | Combination Of Both CGAN And WGAN     |

## BaseModels

[BaseModel](Models/BaseModel.md)

[NaiveBayesBaseModel](Models/NaiveBayesBaseModel.md)

[GradientMethodBaseModel](Models/GradientMethodBaseModel.md)

[IterativeMethodBaseModel](Models/IterativeMethodBaseModel.md)

[DeepReinforcementLearningBaseModel](Models/DeepReinforcementLearningBaseModel.md)

[DeepReinforcementLearningActorCriticBaseModel](Models/DeepReinforcementLearningActorCriticBaseModel.md)

[TabularReinforcementLearningBaseModel](Models/TabularReinforcementLearningBaseModel.md)
