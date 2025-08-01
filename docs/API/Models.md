# [API Reference](../API.md) - Models

## Regression

| Model                                                                                      | Alternate Names | Use Cases                                            |
|--------------------------------------------------------------------------------------------|-----------------|------------------------------------------------------|
| [LinearRegression](Models/LinearRegression.md)                                             | None            | General Price And Time To Level Up Prediction        |
| [NormalLinearRegression](Models/NormalLinearRegression.md) (Not Recommended)               | None            | Final Solution Price And Time To Level Up Prediction |
| [SupportVectorRegression](Models/SupportVectorRegression.md) (May Need Further Refinement) | SVR             | Constrained Price And Time To Level Up Prediction    |
| [KNearestNeighboursRegressor](Models/KNearestNeighboursRegressor.md)                       | KNN-R           | Memory-Based Price And Time To Level Up Prediction   |

## Classification

| Model                                                                                                            | Alternate Names        | Use Cases                                                                                |
|------------------------------------------------------------------------------------------------------------------|------------------------|------------------------------------------------------------------------------------------|
| [KNearestNeighboursClassifier](Models/KNearestNeighboursClassifier.md)                                           | KNN-C                  | Item Recommendation, Similar Player Matchmaking                                          |
| [LogisticRegression](Models/LogisticRegression.md)                                                               | Perceptron             | Purchase Likelihood Estimation, Player Confidence Prediction                             |
| [SupportVectorMachine](Models/SupportVectorMachine.md)                                                           | SVM                    | Hacking Detection, Anomaly Detection                                                     |
| [GaussianNaiveBayes](Models/GaussianNaiveBayes.md)                                                               | None                   | Player Behavior Categorization (e.g. Cautious Vs. Aggressive), Fast State Classification |
| [MultinomialNaiveBayes](Models/MultinomialNaiveBayes.md)                                                         | None                   | Inventory Action Prediction, Strategy Profiling Based on Item Usage                      |
| [BernoulliNaiveBayes](Models/BernoulliNaiveBayes.md)                                                             | None                   | Binary Action Prediction (e.g. Jump Or Not), Quick Decision Filters                      |
| [ComplementNaiveBayes](Models/ComplementNaiveBayes.md)                                                           | None                   | Imbalanced Class Prediction (e.g. Rare Choices, Niche Paths)                             |
| [NeuralNetwork](Models/NeuralNetwork.md)                                                                         | Multi-Layer Perceptron | Decision-Making, Player Behaviour Prediction                                             |

## Clustering

| Model                                                                                                                  | Alternate Names | Use Cases                                                            |
|------------------------------------------------------------------------------------------------------------------------|-----------------|----------------------------------------------------------------------|
| [AffinityPropagation](Models/AffinityPropagation.md)                                                                   | None            | Player Grouping                                                      |
| [AgglomerativeHierarchical](Models/AgglomerativeHierarchical.md)                                                       | None            | Enemy Difficulty Generation                                          |
| [DensityBasedSpatialClusteringOfApplicationsWithNoise](Models/DensityBasedSpatialClusteringOfApplicationsWithNoise.md) | DBSCAN          | Density Grouping                                                     |
| [MeanShift](Models/MeanShift.md)                                                                                       | None            | Boss Spawn Location Search Based On Player Locations                 |
| [ExpectationMaximization](Models/ExpectationMaximization.md)                                                           | EM              | Hacking Detection, Anomaly Detection                                 |
| [KMeans](Models/KMeans.md)                                                                                             | None            | Maximizing Area-of-Effect Abilities, Predictive Target Grouping      |
| [KMedoids](Models/KMedoids.md)                                                                                         | None            | Player Grouping Based On Player Locations With Leader Identification |

## Deep Reinforcement Learning

| Model                                                                                                                            | Alternate Names                           | Use Cases                                                                   |
|----------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------|-----------------------------------------------------------------------------|
| [DeepQLearning](Models/DeepQLearning.md)                                                                                         | Deep Q Network                            | Best Self-Learning Player AIs, Best Recommendation Systems                  |
| [DeepDoubleQLearningV1](Models/DeepDoubleQLearningV1.md)                                                                         | Double Deep Q Network (2010)              | Best Self-Learning Player AIs, Best Recommendation Systems                  |
| [DeepDoubleQLearningV2](Models/DeepDoubleQLearningV2.md)                                                                         | Double Deep Q Network (2015)              | Best Self-Learning Player AIs, Best Recommendation Systems                  |
| [DeepClippedDoubleQLearning](Models/DeepClippedDoubleQLearning.md)                                                               | Clipped Double Deep Q Network             | Best Self-Learning Player AIs, Best Recommendation Systems                  |
| [DeepStateActionRewardStateAction](Models/DeepStateActionRewardStateAction.md)                                                   | Deep SARSA                                | Safe Self-Learning Player AIs, Safe Recommendation Systems                  |
| [DeepDoubleStateActionRewardStateActionV1](Models/DeepDoubleStateActionRewardStateActionV1.md)                                   | Double Deep SARSA                         | Safe Self-Learning Player AIs, Safe Recommendation Systems                  |
| [DeepDoubleStateActionRewardStateActionV2](Models/DeepDoubleStateActionRewardStateActionV2.md)                                   | Double Deep SARSA                         | Safe Self-Learning Player AIs, Safe Recommendation Systems                  |
| [DeepExpectedStateActionRewardStateAction](Models/DeepExpectedStateActionRewardStateAction.md)                                   | Deep Expected SARSA                       | Balanced Self-Learning Player AIs, Balanced Recommendation Systems          |
| [DeepDoubleExpectedStateActionRewardStateActionV1](Models/DeepDoubleExpectedStateActionRewardStateActionV1.md)                   | Double Deep Expected SARSA                | Balanced Self-Learning Player AIs, Balanced Recommendation Systems          |
| [DeepDoubleExpectedStateActionRewardStateActionV2](Models/DeepDoubleExpectedStateActionRewardStateActionV2.md)                   | Double Deep Expected SARSA                | Balanced Self-Learning Player AIs, Balanced Recommendation Systems          |
| [MonteCarloControl](Models/MonteCarloControl.md) (May Need Further Refinement)                                                   | None                                      | Online Self-Learning Player AIs                                             |
| [OffPolicyMonteCarloControl](Models/OffPolicyMonteCarloControl.md)                                                               | None                                      | Offline Self-Learning Player AIs                                            |
| [REINFORCE](Models/REINFORCE.md)                                                                                                 | None                                      | Reward-Based Self-Learning Player AIs                                       |
| [VanillaPolicyGradient](Models/VanillaPolicyGradient.md)                                                                         | VPG                                       | Baseline-Based Self-Learning Player AIs                                     |
| [ActorCritic](Models/ActorCritic.md)                                                                                             | AC                                        | Critic-Based Self-Learning Player AIs                                       |
| [AdvantageActorCritic](Models/AdvantageActorCritic.md)                                                                           | A2C                                       | Advantage-Based Self-Learning Player AIs                                    |
| [ProximalPolicyOptimization](Models/ProximalPolicyOptimization.md)                                                               | PPO                                       | Industry-Grade And Research-Grade Self-Learning Player And Vehicle AIs      |
| [ProximalPolicyOptimizationClip](Models/ProximalPolicyOptimizationClip.md)                                                       | PPO-Clip                                  | Industry-Grade And Research-Grade Self-Learning Player And Vehicle AIs      |
| [SoftActorCritic](Models/SoftActorCritic.md)                                                                                     | SAC                                       | Self-Learning Vehicle AIs                                                   |
| [DeepDeterministicPolicyGradient](Models/DeepDeterministicPolicyGradient.md)                                                     | DDPG                                      | Self-Learning Vehicle AIs                                                   |
| [TwinDelayedDeepDeterministicPolicyGradient](Models/TwinDelayedDeepDeterministicPolicyGradient.md)                               | TD3                                       | Self-Learning Vehicle AIs                                                   |

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
