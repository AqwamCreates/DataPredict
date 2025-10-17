# [API Reference](../API.md) - Models

If you wonder what are the most high-value use cases that helps with retention and revenue generation with this DataPredict™, you can view them [here](../HighValueProjectTutorials.md)!

| Model Type                     | Count |
|--------------------------------|-------|
| Regression                     | 5     |
| Classification                 | 13    |
| Clustering                     | 8     |
| Deep Reinforcement Learning    | 21    |
| Tabular Reinforcement Learning | 12    |
| Generative                     | 4     |
| Total                          | 63    |

For strong deep learning applications, have a look at [DataPredict™ Neural](https://aqwamcreates.github.io/DataPredict-Neural) (object-oriented) and [DataPredict™ Axon](https://aqwamcreates.github.io/DataPredict-Axon) (function-oriented) instead. DataPredict™ is only suitable for general purpose machine, deep and reinforcement learning.

  * Contains most of the deep reinforcement learning and generative algorithms listed here.

  * Includes convolutional, pooling, embedding, dropout and activation layers.

  * Uses reverse-mode automatic differentiation and lazy differentiation evaluation for DataPredict™ Neural (static graph) and DataPredict™ Axon (dynamic graph).

### Note

* Currently, DataPredict™ has ~90% (56 out of 63) models with online learning capabilities. By default, most models would perform offline / batch training on the first train, but then switches to online / incremental / sequential after the first train.

* Tabular reinforcement learning models can use optimizers. And yes, I am quite aware that I have overengineered this, but I really want to make this a grand finale before I stop updating DataPredict™ for a long time.

* No dimensionality reduction algorithms due to not being suitable for game-related use cases. They tend to be computationally expensive and are only useful when a full dataset is collected. This can be offset by choosing proper features and remove the unnecessary ones.

* Going "Gold" on my birthday at 23 January 2026. Probably.

## Regression

| Model                                                                                      | Alternate Names | Use Cases                                                                       |
|--------------------------------------------------------------------------------------------|-----------------|---------------------------------------------------------------------------------|
| [LinearRegression](Models/LinearRegression.md) (Beginner Algorithm)                        | None            | General Time-To-Leave Prediction And In-Game Currency Price Generation          |
| [PassiveAggressiveRegressor](Models/PassiveAggressiveRegressor.md)                         | PA-R            | Fast Constrained Time-To-Leave Prediction And In-Game Currency Price Generation |
| [SupportVectorRegression](Models/SupportVectorRegression.md) (Offline Only)                | SVR             | Constrained Time-To-Leave Prediction And In-Game Currency Price Generation      |
| [KNearestNeighboursRegressor](Models/KNearestNeighboursRegressor.md)                       | KNN-R           | Memory-Based Time-To-Leave Prediction And In-Game Currency Price Generation     |
| [NormalLinearRegression](Models/NormalLinearRegression.md) (Not Recommended)               | None            | Final Solution Time-To-Leave Prediction And In-Game Currency Price Generation   |

## Classification

| Model                                                                                  | Alternate Names        | Use Cases                                                                                                       |
|----------------------------------------------------------------------------------------|------------------------|-----------------------------------------------------------------------------------------------------------------|
| [LogisticRegression](Models/LogisticRegression.md) (Beginner Algorithm)                | Perceptron             | Probability-To-Leave Prediction, Player Churn Prediction, Confidence Prediction                                 |
| [PassiveAggressiveClassifier](Models/PassiveAggressiveClassifier.md)                   | PA-C                   | Fast Purchase Likelihood Estimation, Decision Making                                                            |
| [OneClassPassiveAggressiveClassifier](Models/OneClassPassiveAggressiveClassifier.md)   | OC-PA-C                | Fast Hacking Detection, Anomaly Detection (Using Single Class Data)                                             |
| [NearestCentroid](Models/NearestCentroid.md)                                           | NC                     | Fast Grouping Or Quick Decision Making                                                                          |
| [KNearestNeighboursClassifier](Models/KNearestNeighboursClassifier.md)                 | KNN-C                  | Item Recommendation, Similar Player Matchmaking                                                                 |
| [SupportVectorMachine](Models/SupportVectorMachine.md) (Offline Only)                  | SVM                    | Hacking Detection, Anomaly Detection                                                                            |
| [OneClassSupportVectorMachine](Models/OneClassSupportVectorMachine.md) (Offline Only)  | OC-SVM                 | Hacking Detection, Anomaly Detection (Using Single Class Data)                                                  |
| [NeuralNetwork](Models/NeuralNetwork.md) (Beginner Algorithm)                          | Multi-Layer Perceptron | Decision-Making, Player Behaviour Prediction                                                                    |
| [GaussianNaiveBayes](Models/GaussianNaiveBayes.md) (Stonger As Generative Model)       | None                   | Enemy Data Generation, Player Behavior Categorization (e.g. Cautious Vs. Aggressive), Fast State Classification |
| [MultinomialNaiveBayes](Models/MultinomialNaiveBayes.md) (Stonger As Generative Model) | None                   | Summoning Next Enemy Type, Inventory Action Prediction, Strategy Profiling Based on Item Usage                  |
| [BernoulliNaiveBayes](Models/BernoulliNaiveBayes.md) (Stonger As Generative Model)     | None                   | Binary Action Prediction (e.g. Jump Or Not), Quick Decision Filters                                             |
| [ComplementNaiveBayes](Models/ComplementNaiveBayes.md) (Stonger As Generative Model)   | None                   | Imbalanced Class Prediction (e.g. Rare Choices, Niche Paths)                                                    |
| [CategoricalNaiveBayes](Models/CategoricalNaiveBayes.md) (Stonger As Generative Model) | None                   | Player Choice Prediction (e.g. Weapon Type, Character Class, Map Region Selection)                              |

## Clustering

| Model                                                                                                                                 | Alternate Names | Use Cases                                                            |
|---------------------------------------------------------------------------------------------------------------------------------------|-----------------|----------------------------------------------------------------------|
| [KMeans](Models/KMeans.md) (Beginner Algorithm)                                                                                       | None            | Maximizing Area-of-Effect Abilities, Target Grouping                 |
| [FuzzyCMeans](Models/FuzzyCMeans.md)                                                                                                  | None            | Overlapping Area-of-Effect Abilities, Overlapping Target Grouping    |
| [KMedoids](Models/KMedoids.md)                                                                                                        | None            | Player Grouping Based On Player Locations With Leader Identification |
| [AgglomerativeHierarchical](Models/AgglomerativeHierarchical.md)                                                                      | None            | Enemy Data Generation                                                |
| [ExpectationMaximization](Models/ExpectationMaximization.md)                                                                          | EM              | Hacking Detection, Anomaly Detection                                 |
| [MeanShift](Models/MeanShift.md)                                                                                                      | None            | Boss Spawn Location Search Based On Player Locations                 |
| [AffinityPropagation](Models/AffinityPropagation.md) (Offline Only)                                                                   | None            | Player Grouping                                                      |
| [DensityBasedSpatialClusteringOfApplicationsWithNoise](Models/DensityBasedSpatialClusteringOfApplicationsWithNoise.md) (Offline Only) | DBSCAN          | Density Grouping                                                     |

## Deep Reinforcement Learning

| Model                                                                                                          | Alternate Names               | Use Cases                                                                 |
|----------------------------------------------------------------------------------------------------------------|-------------------------------|---------------------------------------------------------------------------|
| [DeepQLearning](Models/DeepQLearning.md)                                                                       | Deep Q Network                | Best Self-Learning Player AIs, Best Recommendation Systems                |
| [DeepDoubleQLearningV1](Models/DeepDoubleQLearningV1.md)                                                       | Double Deep Q Network (2010)  | Stable Best Self-Learning Player AIs, Best Recommendation Systems         |
| [DeepDoubleQLearningV2](Models/DeepDoubleQLearningV2.md)                                                       | Double Deep Q Network (2015)  | Stable Best Self-Learning Player AIs, Best Recommendation Systems         |
| [DeepClippedDoubleQLearning](Models/DeepClippedDoubleQLearning.md)                                             | Clipped Deep Double Q Network | Stable Best Self-Learning Player AIs, Best Recommendation Systems         |
| [DeepStateActionRewardStateAction](Models/DeepStateActionRewardStateAction.md)                                 | Deep SARSA                    | Safe Self-Learning Player AIs, Safe Recommendation Systems                |
| [DeepDoubleStateActionRewardStateActionV1](Models/DeepDoubleStateActionRewardStateActionV1.md)                 | Double Deep SARSA             | Stable Safe Self-Learning Player AIs, Safe Recommendation Systems         |
| [DeepDoubleStateActionRewardStateActionV2](Models/DeepDoubleStateActionRewardStateActionV2.md)                 | Double Deep SARSA             | Stable Safe Self-Learning Player AIs, Safe Recommendation Systems         |
| [DeepExpectedStateActionRewardStateAction](Models/DeepExpectedStateActionRewardStateAction.md)                 | Deep Expected SARSA           | Balanced Self-Learning Player AIs, Balanced Recommendation Systems        |
| [DeepDoubleExpectedStateActionRewardStateActionV1](Models/DeepDoubleExpectedStateActionRewardStateActionV1.md) | Double Deep Expected SARSA    | Stable Balanced Self-Learning Player AIs, Balanced Recommendation Systems |
| [DeepDoubleExpectedStateActionRewardStateActionV2](Models/DeepDoubleExpectedStateActionRewardStateActionV2.md) | Double Deep Expected SARSA    | Stable Balanced Self-Learning Player AIs, Balanced Recommendation Systems |
| [DeepMonteCarloControl](Models/DeepMonteCarloControl.md) (May Need Further Refinement)                         | None                          | Online Self-Learning Player AIs                                           |
| [DeepOffPolicyMonteCarloControl](Models/DeepOffPolicyMonteCarloControl.md)                                     | None                          | Offline Self-Learning Player AIs                                          |
| [REINFORCE](Models/REINFORCE.md)                                                                               | None                          | Reward-Based Self-Learning Player AIs                                     |
| [VanillaPolicyGradient](Models/VanillaPolicyGradient.md) (May Need Further Refinement)                         | VPG                           | Baseline-Based Self-Learning Player AIs                                   |
| [ActorCritic](Models/ActorCritic.md)                                                                           | AC                            | Critic-Based Self-Learning Player AIs                                     |
| [AdvantageActorCritic](Models/AdvantageActorCritic.md)                                                         | A2C                           | Advantage-Based Self-Learning Player AIs                                  |
| [ProximalPolicyOptimization](Models/ProximalPolicyOptimization.md)                                             | PPO                           | Industry-Grade And Research-Grade Self-Learning Player And Vehicle AIs    |
| [ProximalPolicyOptimizationClip](Models/ProximalPolicyOptimizationClip.md)                                     | PPO-Clip                      | Industry-Grade And Research-Grade Self-Learning Player And Vehicle AIs    |
| [SoftActorCritic](Models/SoftActorCritic.md)                                                                   | SAC                           | Self-Learning Vehicle AIs                                                 |
| [DeepDeterministicPolicyGradient](Models/DeepDeterministicPolicyGradient.md)                                   | DDPG                          | Self-Learning Vehicle AIs                                                 |
| [TwinDelayedDeepDeterministicPolicyGradient](Models/TwinDelayedDeepDeterministicPolicyGradient.md)             | TD3                           | Self-Learning Vehicle AIs                                                 |

## Tabular Reinforcement Learning

| Model                                                                                                                  | Alternate Names           | Use Cases                       |
|------------------------------------------------------------------------------------------------------------------------|---------------------------|---------------------------------|
| [TabularQLearning](Models/TabularQLearning.md)                                                                         | Q-Learning                | Best Self-Learning Grid AIs     |
| [TabularDoubleQLearningV1](Models/TabularDoubleQLearningV1.md)                                                         | Double Q-Learning (2010)  | Best Self-Learning Grid AIs     |
| [TabularDoubleQLearningV2](Models/TabularDoubleQLearningV2.md)                                                         | Double Q-Learning (2015)  | Best Self-Learning Grid AIs     |
| [TabularClippedDoubleQLearning](Models/TabularClippedDoubleQLearning.md)                                               | Clipped Double Q-Learning | Best Self-Learning Grid AIs     |
| [TabularStateActionRewardState](Models/TabularStateActionRewardStateAction.md) (May Need Further Refinement)           | SARSA                     | Safe Self-Learning Grid AIs     |
| [TabularDoubleStateActionRewardStateV1](Models/TabularDoubleStateActionRewardStateV1.md) (May Need Further Refinement) | Double SARSA              | Safe Self-Learning Grid AIs     |
| [TabularDoubleStateActionRewardStateV2](Models/TabularDoubleStateActionRewardStateV2.md) (May Need Further Refinement) | Double SARSA              | Safe Self-Learning Grid AIs     |
| [TabularExpectedStateActionRewardState](Models/TabularExpectedStateActionRewardStateAction.md)                         | Expected SARSA            | Balanced Self-Learning Grid AIs |
| [TabularDoubleExpectedStateActionRewardStateV1](Models/TabularDoubleExpectedStateActionRewardStateV1.md)               | Double Expected SARSA     | Balanced Self-Learning Grid AIs |
| [TabularDoubleExpectedStateActionRewardStateV2](Models/TabularDoubleExpectedStateActionRewardStateV2.md)               | Double Expected SARSA     | Balanced Self-Learning Grid AIs |
| [TabularMonteCarloControl](Models/TabularMonteCarloControl.md)                                                         | MC                        | Online Self-Learning Grid AIs   |
| [TabularOffPolicyMonteCarloControl](Models/TabularOffPolicyMonteCarloControl.md)                                       | Off-Policy MC             | Offline Self-Learning Grid AIs  |

## Generative

| Model                                                                                                              | Alternate Names | Use Cases                                |
|--------------------------------------------------------------------------------------------------------------------|-----------------|------------------------------------------|
| [GenerativeAdversarialNetwork](Models/GenerativeAdversarialNetwork.md)                                             | GAN             | Enemy Data Generation                    |
| [ConditionalGenerativeAdversarialNetwork](Models/ConditionalGenerativeAdversarialNetwork.md)                       | CGAN            | Conditional Enemy Data Generation        |
| [WassersteinGenerativeAdversarialNetwork](Models/WassersteinGenerativeAdversarialNetwork.md)                       | WGAN            | Stable Enemy Data Generation             |
| [ConditionalWassersteinGenerativeAdversarialNetwork](Models/ConditionalWassersteinGenerativeAdversarialNetwork.md) | CWGAN           | Stable Conditional Enemy Data Generation |

## BaseModels

[BaseModel](Models/BaseModel.md)

[NaiveBayesBaseModel](Models/NaiveBayesBaseModel.md)

[GradientMethodBaseModel](Models/GradientMethodBaseModel.md)

[IterativeMethodBaseModel](Models/IterativeMethodBaseModel.md)

[DeepReinforcementLearningBaseModel](Models/DeepReinforcementLearningBaseModel.md)

[DeepReinforcementLearningActorCriticBaseModel](Models/DeepReinforcementLearningActorCriticBaseModel.md)

[TabularReinforcementLearningBaseModel](Models/TabularReinforcementLearningBaseModel.md)
