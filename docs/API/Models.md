# [API Reference](../API.md) - Models

* Remember! This is source-available library and not open-source! You may have a look for more details [here](https://github.com/AqwamCreates/DataPredict/blob/main/docs/DataPredictLibrariesLicensingAgreements.md).

* If you wonder what are the most high-value use cases that helps with revenue and retention generation this DataPredict™, you can view them [here](../HighValueProjectTutorials.md)!

* To see which algorithms that you can swap model parameters with other types of algorithms, you can view them [here](ModelParametersCompatibility.md)!

| Model Type                                                        | Purpose                                         | Count |
|-------------------------------------------------------------------|-------------------------------------------------|-------|
| [Regression](#regression)                                         | Continuous Value Prediction                     | 16    |
| [Classification](#classification)                                 | Feature-Class Prediction                        | 13    |
| [Regression And Classification](#regression-and-classification)   | Continuous Value Or Feature-Class Prediction    | 1     |
| [Clustering](#clustering)                                         | Feature Grouping                                | 10    |
| [Deep Reinforcement Learning](#deep-reinforcement-learning)       | State-Action Optimization Using Neural Networks | 26    |
| [Tabular Reinforcement Learning](#tabular-reinforcement-learning) | State-Action Optimization Using Tables          | 17    |
| [Sequence Modelling](#sequence-modelling)                         | Next State Prediction And Generation            | 3     |
| [Filtering](#filtering)                                           | Next State Tracking / Estimation                | 4     |
| [Outlier Detection](#outlier-detection)                           | Outlier Score Generation                        | 4     |
| [Recommendation](#recommendation)                                 | User-Item Pairing                               | 5     |
| [Generative](#generative)                                         | Feature To Novel Value                          | 4     |
| [Feature-Class Containers](#feature-class-containers)             | Feature-Class Look Up                           | 1     |
| Total                                                             |                                                 | 104   |

### Legend

| Icon | Name                        | Description                                            |
|------|-----------------------------|--------------------------------------------------------|
| ❗   | Implementation Issue       | The model may have some implementation problems.        |
| 🔰   | Beginner Algorithm         | Commonly taught to beginners.                           |
| 💾   | Data Efficient             | Require few data to train the model.                    |
| ⚡   | Computationally Efficient  | Require few computational resources to train the model. |
| 🛡️   | Noise Resistant            | Can handle randomness / unclean data.                   |
| 🟢   | Online                     | Can adapt real-time.                                    |
| 🟡   | Session-Adaptive / Offline | Can be retrained each session.                          |
| ⚠️   | Assumption-Heavy           | Have restrictive rules on using the model.              |
| ⚙️   | Configuration-Heavy        | Requires a lot of manual configuration to use.          |

### Note

* For strong deep learning applications, have a look at [DataPredict™ Neural](https://aqwamcreates.github.io/DataPredict-Neural) (object-oriented, static graph) and [DataPredict™ Axon](https://aqwamcreates.github.io/DataPredict-Axon) (function-oriented, dynamic graph) instead. DataPredict™ is only suitable for general purpose machine, deep and reinforcement learning.

  * Uses reverse-mode automatic differentiation and lazy differentiation evaluation.

  * Includes convolutional, pooling, embedding, dropout and activation layers.

  * Contains most of the deep reinforcement learning and generative algorithms listed here.

* Currently, DataPredict™ has ~93% (95 out of 102) models with online learning capabilities. By default, most models would perform offline / batch training on the first train before switching to online / incremental / sequential after the first train.

* No dimensionality reduction algorithms due to not being suitable for game-related use cases. They tend to be computationally expensive and are only useful when a full dataset is collected. This can be offset by choosing proper features and remove the unnecessary ones.

* No tree models (like decision trees) for now due to these models requiring the full dataset and tend to be computationally expensive. In addition, most of these tree models do not have online learning capabilities.

## Regression

> ❗Implementation Issue 🔰 Beginner Algorithm 💾 Data Efficient ⚡ Computationally Efficient 🛡️ Noise Resistant 🟢 Online 🟡 Session-Adaptive / Offline ⚠️ Assumption-Heavy ⚙️ Configuration-Heavy

| Model                                                                                      | Alternate Names | Properties        | Use Cases                                                                                                |
|--------------------------------------------------------------------------------------------|-----------------|-------------------|----------------------------------------------------------------------------------------------------------|
| [LinearRegression](Models/LinearRegression.md)                                             | LR              | 🔰 🟢 🟡        | General Time-To-Leave Prediction And In-Game Currency Price Generation                                   |
| [QuantileRegression](Models/QuantileRegression.md)                                         | None            | 🟢 🟡            | Case-Based Time-To-Leave Prediction And In-Game Currency Price Generation                                |
| [PoissonRegression](Models/PoissonRegression.md)                                           | None            | 🟢 🟡 ⚠️         | Positive-Integer-Based Time-To-Leave Prediction And In-Game Currency Price Generation                    |
| [NegativeBinomialRegression](Models/NegativeBinomialRegression.md)                         | None            | 🟢 🟡 ⚠️         | Positive-Integer-Based Time-To-Leave Prediction And In-Game Currency Price Generation                    |
| [GammaRegression](Models/GammaRegression.md)                                               | None            | ❗ 🟢 🟡 ⚠️      | Player Session Duration Prediction And Content Engagement Time Prediction                               |
| [IsotonicRegression](Models/IsotonicRegression.md)                                         | None            | ⚡ 🟢 🟡         | 1-Dimensional Skill-Based Time-To-Leave Prediction                                                      |
| [PassiveAggressiveRegressor](Models/PassiveAggressiveRegressor.md)                         | PA-R            | ⚡ 🟢            | Fast Constrained Time-To-Leave Prediction And In-Game Currency Price Generation                          |
| [SupportVectorRegression](Models/SupportVectorRegression.md)                               | SVR             | 💾 🟡            | Constrained Time-To-Leave Prediction And In-Game Currency Price Generation                               |
| [SupportVectorRegressionGradientVariant](Models/SupportVectorRegressionGradientVariant.md) | SVR             | 🟢 🟡            | Real-Time Constrained Time-To-Leave Prediction And In-Game Currency Price Generation                     |
| [KNearestNeighboursRegressor](Models/KNearestNeighboursRegressor.md)                       | KNN-R           | 🟢 🟡            | Memory-Based Time-To-Leave Prediction And In-Game Currency Price Generation                              |
| [OrdinaryLeastSquaresRegression](Models/OrdinaryLeastSquaresRegression.md)*                | None            | 💾 ⚡ 🟢 🟡 ⚠️ | Instant Train Time-To-Leave Prediction And In-Game Currency Price Generation                             |
| [WeightedLeastSquaresRegression](Models/WeightedLeastSquaresRegression.md)*                | None            | 💾 ⚡ 🟢 🟡 ⚠️ | Instant Train Time-To-Leave Prediction And In-Game Currency Price Generation With Probability Estimation |
| [RidgeRegression](Models/RidgeRegression.md)*                                              | None            | 💾 ⚡ 🟢 🟡 ⚠️ | Instant Train Time-To-Leave Prediction And In-Game Currency Price Generation                             |
| [BayesianLinearRegression](Models/BayesianLinearRegression.md)*                            | None            | 💾 ⚡ 🟢 🟡 ⚠️ | Instant Train Time-To-Leave Prediction And In-Game Currency Price Generation With Probability Estimation |
| [BayesianQuantileLinearRegression](Models/BayesianQuantileLinearRegression.md)*            | None            | 💾 ⚡ 🟢 🟡 ⚠️ | Instant Train Time-To-Leave Prediction And In-Game Currency Price Generation With Case Estimation        |
| [RecursiveLeastSquaresRegression](Models/RecursiveLeastSquaresRegression.md)               | RLS             | 💾 ⚡ 🟢 🟡     | Instant Train Time-To-Leave Prediction And In-Game Currency Price Generation With Probability Estimation |

\* The moodels assume that the features have a linear relationship with the label values, which is almost certainly not true in game-related settings.

## Classification

> ❗Implementation Issue 🔰 Beginner Algorithm 💾 Data Efficient ⚡ Computationally Efficient 🛡️ Noise Resistant 🟢 Online 🟡 Session-Adaptive / Offline ⚠️ Assumption-Heavy ⚙️ Configuration-Heavy

| Model                                                                                | Alternate Names                | Properties       | Use Cases                                                                                                      |
|--------------------------------------------------------------------------------------|--------------------------------|------------------|----------------------------------------------------------------------------------------------------------------|
| [BinaryRegression](Models/BinaryRegression.md)                                       | Perceptron, Sigmoid Regression | 🔰 🟢 🟡       | Probability-To-Leave Prediction, Player Churn Prediction, Confidence Prediction                                 |
| [PassiveAggressiveClassifier](Models/PassiveAggressiveClassifier.md)                 | PA-C                           | ⚡ 🟢           | Fast Purchase Likelihood Estimation, Decision Making                                                            |
| [NearestCentroid](Models/NearestCentroid.md)                                         | NC                             | ⚡ 🟢 🟡        | Fast Grouping Or Quick Decision Making                                                                         |
| [KNearestNeighboursClassifier](Models/KNearestNeighboursClassifier.md)               | KNN-C                          | 🟢 🟡           | Item Recommendation, Similar Player Matchmaking                                                                 |
| [SupportVectorMachine](Models/SupportVectorMachine.md)                               | SVM                            | 💾 🟡          | Boundary-Based Prediction                                                                                        |
| [SupportVectorMachineGradientVariant](Models/SupportVectorMachineGradientVariant.md) | SVM                            | 🟢 🟡          | Real-Time Boundary-Based Prediction                                                                              |
| [NeuralNetwork](Models/NeuralNetwork.md)                                             | Multi-Layer Perceptron         | 🟢 🟡           | Decision-Making, Player Behaviour Prediction                                                                    |
| [GaussianNaiveBayes](Models/GaussianNaiveBayes.md)*                                  | GNB                            | 💾 ⚡ 🟢 🟡 ⚠️ | Enemy Data Generation, Player Behavior Categorization (e.g. Cautious Vs. Aggressive), Fast State Classification |
| [MultinomialNaiveBayes](Models/MultinomialNaiveBayes.md)*                            | MNB                            | 💾 ⚡ 🟢 🟡 ⚠️ |Summoning Next Enemy Type, Inventory Action Prediction, Strategy Profiling Based on Item Usage                   |
| [BernoulliNaiveBayes](Models/BernoulliNaiveBayes.md)*                                | BNB                            | 💾 ⚡ 🟢 🟡 ⚠️ | Binary Action Prediction (e.g. Jump Or Not), Quick Decision Filters                                             |
| [ComplementNaiveBayes](Models/ComplementNaiveBayes.md)*                              | CNB                            | 💾 ⚡ 🟢 🟡 ⚠️ | Imbalanced Class Prediction (e.g. Rare Choices, Rare Paths)                                                     |
| [CategoricalNaiveBayes](Models/CategoricalNaiveBayes.md)*                            | CNB                            | 💾 ⚡ 🟢 🟡 ⚠️ | Player Choice Prediction (e.g. Weapon Type, Character Class, Map Region Selection)                              |
| [OrdinalRegression](Models/OrdinalRegression.md)                                     | Ordinal Classification         | 🟢 🟡 ⚠️        | Skill Tier Prediction, Dynamic Difficulty Adjustment, Ranking Systems                                           |

\* "Naive Bayes" models assumes that the features are independent to each other, which is almost certainly not true in game-related settings. Additionally, these models are better as generative models, despite being commonly taught as a classifier.

## Regression And Classification

> ❗Implementation Issue 🔰 Beginner Algorithm 💾 Data Efficient ⚡ Computationally Efficient 🛡️ Noise Resistant 🟢 Online 🟡 Session-Adaptive / Offline ⚠️ Assumption-Heavy ⚙️ Configuration-Heavy

| Model                                                                                            | Alternate Names | Properties     | Use Cases                                                               |
|--------------------------------------------------------------------------------------------------|-----------------|----------------|-------------------------------------------------------------------------|
| [IterativeReweightedLeastSquaresRegression](Models/IterativeReweightedLeastSquaresRegression.md) | IRLS            |❗ 🔰 🟢 🟡 ⚡| General Time-To-Leave Prediction And In-Game Currency Price Generation   |

## Clustering

> ❗Implementation Issue 🔰 Beginner Algorithm 💾 Data Efficient ⚡ Computationally Efficient 🛡️ Noise Resistant 🟢 Online 🟡 Session-Adaptive / Offline ⚠️ Assumption-Heavy ⚙️ Configuration-Heavy

| Model                                                                                                                  | Alternate Names | Properties | Use Cases                                                            |
|------------------------------------------------------------------------------------------------------------------------|-----------------|------------|----------------------------------------------------------------------|
| [KMeans](Models/KMeans.md)                                                                                             | None            | 🔰 🟢 🟡  | Maximizing Area-of-Effect Abilities, Maximizing Target Grouping      |
| [FuzzyCMeans](Models/FuzzyCMeans.md)                                                                                   | None            | 🟢 🟡     | Overlapping Area-of-Effect Abilities, Overlapping Target Grouping    |
| [KMedoids](Models/KMedoids.md)                                                                                         | None            | 🟢 🟡     | Player Grouping Based On Player Locations With Leader Identification |
| [AgglomerativeHierarchical](Models/AgglomerativeHierarchical.md)                                                       | None            | 🟢 🟡     | Enemy Data Generation                                                |
| [ExpectationMaximization](Models/ExpectationMaximization.md)                                                           | EM              | 🟢 🟡     | Hacking Detection, Anomaly Detection                                 |
| [MeanShift](Models/MeanShift.md)                                                                                       | None            | 🛡️ 🟢 🟡 | Boss Spawn Location Search Based On Player Locations                 |
| [AffinityPropagation](Models/AffinityPropagation.md)                                                                   | AP              | 🟡        | Player Grouping                                                      |
| [DensityBasedSpatialClusteringOfApplicationsWithNoise](Models/DensityBasedSpatialClusteringOfApplicationsWithNoise.md) | DBSCAN          | 🛡️ 🟡     | Density Grouping                                                     |
| [OrderingPointsToIdentifyClusteringStructure](Models/OrderingPointsToIdentifyClusteringStructure.md)                   | OPTICS          | 🛡️ 🟡     | Density Grouping                                                     |
| [BisectingCluster](Models/BisectingCluster.md)                                                                         | None            | ⚡ 🟡     | Slow To Quick Grouping                                               |

## Deep Reinforcement Learning

> ❗Implementation Issue 🔰 Beginner Algorithm 💾 Data Efficient ⚡ Computationally Efficient 🛡️ Noise Resistant 🟢 Online 🟡 Session-Adaptive / Offline ⚠️ Assumption-Heavy ⚙️ Configuration-Heavy

| Model                                                                                                          | Alternate Names               | Properties  | Use Cases                                                                 |
|----------------------------------------------------------------------------------------------------------------|-------------------------------|-------------|---------------------------------------------------------------------------|
| [DeepQLearning](Models/DeepQLearning.md)                                                                       | Deep Q Network                | 💾 🟢      | Best Self-Learning Player AIs, Best Recommendation Systems                |
| [DeepNStepQLearning](Models/DeepNStepQLearning.md)                                                             | Deep N-Step Q Network         | 💾 🟢      | Best Self-Learning Player AIs, Best Recommendation Systems                |
| [DeepDoubleQLearningV1](Models/DeepDoubleQLearningV1.md)                                                       | Double Deep Q Network (2010)  | 💾 🛡️ 🟢   | Stable Best Self-Learning Player AIs, Best Recommendation Systems         |
| [DeepDoubleQLearningV2](Models/DeepDoubleQLearningV2.md)                                                       | Double Deep Q Network (2015)  | 💾 🛡️ 🟢   | Stable Best Self-Learning Player AIs, Best Recommendation Systems         |
| [DeepClippedDoubleQLearning](Models/DeepClippedDoubleQLearning.md)                                             | Clipped Deep Double Q Network | 💾 🛡️ 🟢   | Stable Best Self-Learning Player AIs, Best Recommendation Systems         |
| [DeepStateActionRewardStateAction](Models/DeepStateActionRewardStateAction.md)                                 | Deep SARSA                    | 🟢          | Safe Self-Learning Player AIs, Safe Recommendation Systems                |
| [DeepNStepStateActionRewardStateAction](Models/DeepNStepStateActionRewardStateAction.md)                       | Deep N-Step SARSA             | 🟢          | Safe Self-Learning Player AIs, Safe Recommendation Systems                |
| [DeepDoubleStateActionRewardStateActionV1](Models/DeepDoubleStateActionRewardStateActionV1.md)                 | Double Deep SARSA             | 🛡️ 🟢      | Stable Safe Self-Learning Player AIs, Safe Recommendation Systems         |
| [DeepDoubleStateActionRewardStateActionV2](Models/DeepDoubleStateActionRewardStateActionV2.md)                 | Double Deep SARSA             | 🛡️ 🟢      | Stable Safe Self-Learning Player AIs, Safe Recommendation Systems         |
| [DeepExpectedStateActionRewardStateAction](Models/DeepExpectedStateActionRewardStateAction.md)                 | Deep Expected SARSA           | 🟢         | Balanced Self-Learning Player AIs, Balanced Recommendation Systems        |
| [DeepNStepExpectedStateActionRewardStateAction](Models/DeepExpectedStateActionRewardStateAction.md)            | Deep N-Step Expected SARSA    | 🟢         | Balanced Self-Learning Player AIs, Balanced Recommendation Systems        |
| [DeepDoubleExpectedStateActionRewardStateActionV1](Models/DeepDoubleExpectedStateActionRewardStateActionV1.md) | Double Deep Expected SARSA    | 🛡️ 🟢      | Stable Balanced Self-Learning Player AIs, Balanced Recommendation Systems |
| [DeepDoubleExpectedStateActionRewardStateActionV2](Models/DeepDoubleExpectedStateActionRewardStateActionV2.md) | Double Deep Expected SARSA    | 🛡️ 🟢      | Stable Balanced Self-Learning Player AIs, Balanced Recommendation Systems |
| [DeepMonteCarloControl](Models/DeepMonteCarloControl.md)                                                       | None                          | ❗ 🟢      | Online Self-Learning Player AIs                                           |
| [DeepOffPolicyMonteCarloControl](Models/DeepOffPolicyMonteCarloControl.md)                                     | None                          | 🟢         | Offline Self-Learning Player AIs                                          |
| [DeepTemporalDifference](Models/DeepTemporalDifference.md)                                                     | TD                            | 🟢         | Priority Systems                                                          |
| [DeepREINFORCE](Models/DeepREINFORCE.md)                                                                       | None                          | 🟢         | Reward-Based Self-Learning Player AIs                                     |
| [VanillaPolicyGradient](Models/VanillaPolicyGradient.md)                                                       | VPG                           | ❗ 🟢      | Baseline-Based Self-Learning Player AIs                                   |
| [ActorCritic](Models/ActorCritic.md)                                                                           | AC                            | 🟢         | Critic-Based Self-Learning Player AIs                                     |
| [AdvantageActorCritic](Models/AdvantageActorCritic.md)                                                         | A2C                           | 🟢         | Advantage-Based Self-Learning Player AIs                                  |
| [TemporalDifferenceActorCritic](Models/TemporalDifferenceActorCritic.md)                                       | TD-AC                         | 🟢         | Bootsrapped Online Self-Learning Player AIs                               |
| [ProximalPolicyOptimization](Models/ProximalPolicyOptimization.md)                                             | PPO                           | 🟢         | Industry-Grade And Research-Grade Self-Learning Player And Vehicle AIs    |
| [ProximalPolicyOptimizationClip](Models/ProximalPolicyOptimizationClip.md)                                     | PPO-Clip                      | 🟢         | Industry-Grade And Research-Grade Self-Learning Player And Vehicle AIs    |
| [SoftActorCritic](Models/SoftActorCritic.md)                                                                   | SAC                           | 💾 🛡️ 🟢  | Self-Learning Vehicle AIs                                                 |
| [DeepDeterministicPolicyGradient](Models/DeepDeterministicPolicyGradient.md)                                   | DDPG                          | 🟢 ⚙️      | Self-Learning Vehicle AIs                                                 |
| [TwinDelayedDeepDeterministicPolicyGradient](Models/TwinDelayedDeepDeterministicPolicyGradient.md)             | TD3                           | 🟢 🛡️ ⚙️   | Self-Learning Vehicle AIs                                                 |

## Tabular Reinforcement Learning

> ❗Implementation Issue 🔰 Beginner Algorithm 💾 Data Efficient ⚡ Computationally Efficient 🛡️ Noise Resistant 🟢 Online 🟡 Session-Adaptive / Offline ⚠️ Assumption-Heavy ⚙️ Configuration-Heavy

| Model                                                                                                                | Alternate Names           | Properties  | Use Cases                           |
|----------------------------------------------------------------------------------------------------------------------|---------------------------|-------------|-------------------------------------|
| [TabularQLearning](Models/TabularQLearning.md)                                                                       | Q-Learning                | 🔰 💾 🟢   | Best Self-Learning Grid AIs        |
| [TabularNStepQLearning](Models/TabularNStepQLearning.md)                                                             | N-Step Q-Learning         | 🔰 💾 🟢   | Best Self-Learning Grid AIs        |
| [TabularDoubleQLearningV1](Models/TabularDoubleQLearningV1.md)                                                       | Double Q-Learning (2010)  | 💾 🛡️ 🟢   | Best Self-Learning Grid AIs        |
| [TabularDoubleQLearningV2](Models/TabularDoubleQLearningV2.md)                                                       | Double Q-Learning (2015)  | 💾 🛡️ 🟢   | Best Self-Learning Grid AIs        |
| [TabularClippedDoubleQLearning](Models/TabularClippedDoubleQLearning.md)                                             | Clipped Double Q-Learning | 💾 🛡️ 🟢   | Best Self-Learning Grid AIs        |
| [TabularStateActionRewardStateAction](Models/TabularStateActionRewardStateAction.md)                                 | SARSA                     | 🔰 🟢       | Safe Self-Learning Grid AIs        |
| [TabularNStepStateActionRewardStateAction](Models/TabularNStepStateActionRewardStateAction.md)                       | N-Step SARSA              | 🔰 🟢       | Safe Self-Learning Grid AIs        |
| [TabularDoubleStateActionRewardStateActionV1](Models/TabularDoubleStateActionRewardStateActionV1.md)                 | Double SARSA              | 🛡️ 🟢       | Safe Self-Learning Grid AIs        |
| [TabularDoubleStateActionRewardStateActionV2](Models/TabularDoubleStateActionRewardStateActionV2.md)                 | Double SARSA              | 🛡️ 🟢       | Safe Self-Learning Grid AIs        |
| [TabularExpectedStateActionRewardStateAction](Models/TabularExpectedStateActionRewardStateAction.md)                 | Expected SARSA            | 🟢          | Balanced Self-Learning Grid AIs     |
| [TabularNStepExpectedStateActionRewardStateAction](Models/TabularNStepExpectedStateActionRewardStateAction.md)       | N-Step Expected SARSA     | 🟢          | Balanced Self-Learning Grid AIs     |
| [TabularDoubleExpectedStateActionRewardStateActionV1](Models/TabularDoubleExpectedStateActionRewardStateActionV1.md) | Double Expected SARSA     | 🛡️ 🟢       | Balanced Self-Learning Grid AIs     |
| [TabularDoubleExpectedStateActionRewardStateActionV2](Models/TabularDoubleExpectedStateActionRewardStateActionV2.md) | Double Expected SARSA     | 🛡️ 🟢       | Balanced Self-Learning Grid AIs     |
| [TabularMonteCarloControl](Models/TabularMonteCarloControl.md)                                                       | MC                        | 🟢          | Online Self-Learning Grid AIs       |
| [TabularOffPolicyMonteCarloControl](Models/TabularOffPolicyMonteCarloControl.md)                                     | Off-Policy MC             | 🟢          | Offline Self-Learning Grid AIs      |
| [TabularTemporalDifference](Models/TabularTemporalDifference.md)                                                     | TD                        | 🟢          | Priority Systems                    |
| [TabularREINFORCE](Models/TabularREINFORCE.md)                                                                       | None                      | 🟢          | Reward-Based Self-Learning Grid AIs |

## Sequence Modelling

> ❗Implementation Issue 🔰 Beginner Algorithm 💾 Data Efficient ⚡ Computationally Efficient 🛡️ Noise Resistant 🟢 Online 🟡 Session-Adaptive / Offline ⚠️ Assumption-Heavy ⚙️ Configuration-Heavy

| Model                                                         | Alternate Names | Properties | Use Cases                                 |
|---------------------------------------------------------------|-----------------|------------|-------------------------------------------|
| [Markov](Models/Markov.md)*                                   | None            | 💾 🟢     | Single Feature Player State Prediction    |
| [DynamicBayesianNetwork](Models/DynamicBayesianNetwork.md)*   | DBN             | 💾 🟢     | Multiple Features Player State Prediction |
| [ConditionalRandomField](Models/ConditionalRandomField.md)*   | CRF             | 🟢         | Multiple Features Player State Prediction |

* These are single step variants of the sequence models. Hence, it will not use or return sequence of values.

## Filtering

> ❗Implementation Issue 🔰 Beginner Algorithm 💾 Data Efficient ⚡ Computationally Efficient 🛡️ Noise Resistant 🟢 Online 🟡 Session-Adaptive / Offline ⚠️ Assumption-Heavy ⚙️ Configuration-Heavy

| Model                                                                                            | Alternate Names | Properties | Use Cases                       |
|--------------------------------------------------------------------------------------------------|-----------------|------------|---------------------------------|
| [KalmanFilter](Models/KalmanFilter.md)                                                           | KF              | 🟢 ⚠️     | Linear Movement Anti-Cheat     |
| [ExtendedKalmanFilter](Models/ExtendedKalmanFilter.md)                                           | EKF             | 🟢 ⚙️     | Non-Linear Movement Anti-Cheat |
| [UnscentedKalmanFilter](Models/UnscentedKalmanFilter.md)                                         | UKF             | 💾 🟢 ⚙️  | Non-Linear Movement Anti-Cheat |
| [UnscentedKalmanFilter (DataPredict Variant)](Models/UnscentedKalmanFilterDataPredictVariant.md) | UKF-DP          | 💾 🟢 ⚙️  | Non-Linear Movement Anti-Cheat |

## Outlier Detection

> ❗Implementation Issue 🔰 Beginner Algorithm 💾 Data Efficient ⚡ Computationally Efficient 🛡️ Noise Resistant 🟢 Online 🟡 Session-Adaptive / Offline ⚠️ Assumption-Heavy ⚙️ Configuration-Heavy

| Model                                                                                | Alternate Names | Properties | Use Cases                                                           |
|--------------------------------------------------------------------------------------|-----------------|------------| --------------------------------------------------------------------|
| [OneClassSupportVectorMachine](Models/OneClassSupportVectorMachine.md)               | OC-SVM          | 💾 🟡     | Hacking Detection, Anomaly Detection (Using Single Class Data)      |
| [OneClassPassiveAggressiveClassifier](Models/OneClassPassiveAggressiveClassifier.md) | OC-PA-C         | ❗ ⚡ 🟢  | Fast Hacking Detection, Anomaly Detection (Using Single Class Data) |
| [LocalOutlierFactor](Models/LocalOutlierFactor.md)                                   | LOF             | 🟢 🟡     | Score-Based Play-Time Milestone Detection                           |
| [LocalOutlierProbability](Models/LocalOutlierProbability.md)                         | LoOP            | 🟢 🟡     | Probability-Based Play-Time Milestone Detection                     |

## Recommendation

> ❗Implementation Issue 🔰 Beginner Algorithm 💾 Data Efficient ⚡ Computationally Efficient 🛡️ Noise Resistant 🟢 Online 🟡 Session-Adaptive / Offline ⚠️ Assumption-Heavy ⚙️ Configuration-Heavy

| Model                                                                                      | Alternate Names | Properties | Use Cases                                    |
|--------------------------------------------------------------------------------------------|-----------------|------------| ---------------------------------------------|
| [FactorizationMachine](Models/FactorizationMachine.md)                                     | FM              | 🟢 🟡 🛡️  | Cold-Start User-Item Recommendation          |
| [FactorizedPairwiseInteraction](Models/FactorizedPairwiseInteraction.md)                   | None            | 🟢 🟡 🛡️  | Cold-Start User-Item Recommendation          |
| [SimonFunkMatrixFactorization](Models/SimonFunkMatrixFactorization.md)                     | Funk MF         | 🟢 🟡     | Early Netflix-Style User-Item Recommendation |
| [SimonFunkMatrixFactorizationWithBiases](Models/SimonFunkMatrixFactorizationWithBiases.md) | Funk MF         | 🟢 🟡 🛡️  | Early Netflix-Style User-Item Recommendation |
| [TwoTower](Models/TwoTower.md)                                                             | None            | 🟢 🟡     | YouTube-Style User-Item Recommendation        |

## Generative

> ❗Implementation Issue 🔰 Beginner Algorithm 💾 Data Efficient ⚡ Computationally Efficient 🛡️ Noise Resistant 🟢 Online 🟡 Session-Adaptive / Offline ⚠️ Assumption-Heavy ⚙️ Configuration-Heavy

| Model                                                                                                              | Alternate Names | Properties | Use Cases                                 |
|--------------------------------------------------------------------------------------------------------------------|-----------------|------------| ------------------------------------------|
| [GenerativeAdversarialNetwork](Models/GenerativeAdversarialNetwork.md)                                             | GAN             | 🟢 🟡     | Enemy Data Generation                     |
| [ConditionalGenerativeAdversarialNetwork](Models/ConditionalGenerativeAdversarialNetwork.md)                       | CGAN            | 🟢 🟡     | Conditional Enemy Data Generation         |
| [WassersteinGenerativeAdversarialNetwork](Models/WassersteinGenerativeAdversarialNetwork.md)                       | WGAN            | 🟢 🟡     | Stable Enemy Data Generation              |
| [ConditionalWassersteinGenerativeAdversarialNetwork](Models/ConditionalWassersteinGenerativeAdversarialNetwork.md) | CWGAN           | 🟢 🟡     | Stable Conditional Enemy Data Generation  |

## Feature-Class Containers

> ❗Implementation Issue 🔰 Beginner Algorithm 💾 Data Efficient ⚡ Computationally Efficient 🛡️ Noise Resistant 🟢 Online 🟡 Session-Adaptive / Offline ⚠️ Assumption-Heavy ⚙️ Configuration-Heavy

| Model                    | Alternate Names | Properties | Use Cases      |
|--------------------------|-----------------|------------| ---------------|
| [Table](Models/Table.md) | Grid            | ⚡🟢 🟡   | Quick Look Up  |

## BaseModels

[BaseModel](Models/BaseModel.md)

[NaiveBayesBaseModel](Models/NaiveBayesBaseModel.md)

[IterativeMethodBaseModel](Models/IterativeMethodBaseModel.md)

[GradientMethodBaseModel](Models/GradientMethodBaseModel.md)

[GenerativeAdversarialNetworkBaseModel](GenerativeAdversarialNetworkBaseModel.md)

[MatrixFactorizationBaseModel](Models/MatrixFactorizationBaseModel.md)

[TabularReinforcementLearningBaseModel](Models/TabularReinforcementLearningBaseModel.md)

[DeepReinforcementLearningBaseModel](Models/DeepReinforcementLearningBaseModel.md)

[DeepReinforcementLearningActorCriticBaseModel](Models/DeepReinforcementLearningActorCriticBaseModel.md)
