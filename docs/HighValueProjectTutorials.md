# High-Value Project Tutorials

### Disclaimer

* References that validates the use cases can be found [here](HighValueProjectTutorials/References.md). It also includes my papers.

* The "minimal implementation time" assumes that a junior gameplay machine learning engineer is handling the implementation.

* Since DataPredict is written in native Lua, you can have extra compute per player alongside a single Roblox server by loading the models on players' Roblox client.

  * Phone users: Likely have 4 GB - 8 GB RAM. Variable CPU.
 
  * PC users: Likely have 8 GB - 16 GB RAM. Variable CPU.

* Before you engage in integrating machine, deep and reinforcement learning models into live projects, I recommend you to have a look at safe practices [here](HighValueProjectTutorials/SafePracticesForLiveProjects.md).

* The content of this page and its links are licensed under the DataPredict™ library's [Terms And Conditions](TermsAndConditions.md). This includes the codes shown in the links below.

  * Therefore, creating or redistributing copies or derivatives of this page and its links' contents are not allowed.

  * Commercial use is also not allowed without a license (except certain conditions).

* You can download and read the full list of commercial licensing agreements [here](https://github.com/AqwamCreates/DataPredict/blob/main/docs/DataPredictLibrariesLicensingAgreements.md).

* For information regarding potential license violations and eligibility for a bounty reward, please refer to the [Terms And Conditions Violation Bounty Reward Information](TermsAndConditionsViolationBountyRewardInformation.md).

## Retention Systems

* [Creating Time-To-Leave Prediction Model](HighValueProjectTutorials/CreatingTimeToLeavePredictionModel.md)

  * No need to add new content; the model can use existing content to optimize your games.

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

* [Creating Probability-To-Leave Prediction Model](HighValueProjectTutorials/CreatingProbabilityToLeavePredictionModel.md)

  * No need to add new content; the model can use existing content to optimize your games.

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

* [Creating Left-Too-Early Detection Model](HighValueProjectTutorials/CreatingLeftTooEarlyDetectionModel.md)

   * Inverse of probability-to-leave model by detecting outliers.

  * No need to add new content; the model can use existing content to optimize your games.

   * Highly exploitable if the player accumulates long session times over many sessions before suddenly decrease the session times gradually if rewards are involved.

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

* [Creating Labelless Left-Too-Early Detection Model](HighValueProjectTutorials/CreatingLabellessLeftTooEarlyDetectionModel.md)

  * Same as "Left-Too-Early Detection Model", but it does not require manual tracking of label data, which makes it less accurate.

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

* [Creating Play Time Maximization Model](HighValueProjectTutorials/CreatingPlayTimeMaximizationModel.md)

  * The model chooses actions or events that maximizes play time.

  * No need to add new content; the model can use existing content to optimize your games.

  * Have higher play time potential due to its ability to exploit and explore than the other four models, but tend to be risky to use.

  * Minimal implementation takes a minimum of 2 hours using DataPredict™, especially if custom events are associated with the model's output.

* [Creating Engagement Milestone Detection Ensemble Model](HighValueProjectTutorials/CreatingEngagementMilestoneDetectionEnsembleModel.md)

  * Uses a combination of:
 
    * Time-To-Leave Prediction Model
   
    * Probability-To-Leave Prediction Model
   
    * Creating Left-Too-Early Detection Model

  * The model periodically checks if the player is playing much more longer or more engaged than usual.

  * Minimal implementation takes a minimum of 4 hours using DataPredict™.

* [Creating Play Time Maximization Ensemble Model](HighValueProjectTutorials/CreatingPlayTimeMaximizationEnsembleModel.md)

  * Uses a combination of:
 
    * Time-To-Leave Prediction Model
   
    * Probability-To-Leave Prediction Model
   
    * Play Time Maximization Model

  * Less risky than the original "Play Time Maximization Model", but takes more time to implement.

  * Minimal implementation takes a minimum of 4 hours using DataPredict™.

## Regression Recommendation Systems

* [Creating Probability-Based Recommendation Model](HighValueProjectTutorials/CreatingProbabilityBasedRecommendationModel.md)

  * Minimal implementation takes a minimum of 2 hours using DataPredict™.

* [Creating Similarity-Based Recommendation Model](HighValueProjectTutorials/CreatingSimilarityBasedRecommendationModel.md)

  * Memory-based model. May eat up storage space.

  * Minimal implementation takes a minimum of 2 hours using DataPredict™.

* [Creating Reward-Maximization-Based Regression Recommendation Model](HighValueProjectTutorials/CreatingRewardMaximizationBasedRegressionRecommendationModel.md)

  * Limited to one recommendation at a time.

  * Have higher monetization potential due to its ability to exploit and explore than the other two models, but tend to be risky to use.

  * Minimal implementation takes a minimum of 2 hours using DataPredict™, especially if multiple recommendations are made.

## Binary Recommendation Systems

* [Creating Classification-Based Recommendation Model](HighValueProjectTutorials/CreatingClassificationBasedRecommendationModel.md)

  * Minimal implementation takes a minimum of 2 hours using DataPredict™.

* [Creating Reward-Maximization-Based Binary Recommendation Model](HighValueProjectTutorials/CreatingRewardMaximizationBasedBinaryRecommendationModel.md)

  * Limited to one recommendation at a time.

  * Have higher monetization potential due to its ability to exploit and explore than the classification-based model, but tend to be risky to use.

  * Minimal implementation takes a minimum of 2 hours using DataPredict™, especially if multiple recommendations are made.

## Dynamic Difficulty Adjustment Systems

* [Creating Regression-Based Enemy Data Generation Model](HighValueProjectTutorials/CreatingRegressionBasedEnemyDataGenerationModel.md)

  * Every time a player kills an enemy, both player's combat data and enemy's data are used to train the model.

  * No need to add new content; the model can use existing content to optimize your games.

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

* [Creating Probability-Based Enemy Data Generation Model](HighValueProjectTutorials/CreatingProbabilityBasedEnemyDataGenerationModel.md)

  * Uses players' combat data (optional) paired with enemies' data to predict likelihood that the player will engage with the enemy if spawned.

  * Uses "Generative One Class Support Vector Machine", which is an unconventional generative method that probably don't exist in the research literature.

  * No need to add new content; the model can use existing content to optimize your games.

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

* [Creating Cluster-Based Enemy Data Generation Multiplayer Model](HighValueProjectTutorials/CreatingClusterBasedEnemyDataGenerationForMultiplayerModel.md)

  * Uses players' combat data to generate the center of enemies' data, creating difficulty suited for the majority of players in a session.

  * No need to add new content; the model can use existing content to optimize your games.

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

* [Creating Cluster-Based Enemy Data Generation Singleplayer Model](HighValueProjectTutorials/CreatingClusterBasedEnemyDataGenerationForSingleplayerModel.md)

  * Uses a single player's defeated enemies' data to generate the center of enemies' data, creating difficulty personal to that player.

  * No need to add new content; the model can use existing content to optimize your games.

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

* [Creating Unconditional-Diversity-Based Enemy Data Generation Model](HighValueProjectTutorials/CreatingUnconditionalDiversityBasedEnemyDataGenerationModel.md)

  * Uses enemies' data as generator's output. The generated enemy data is then used against the discriminator to see if the player will interact with it.
 
    * Think of the discriminator as "fake player" and the discriminator will see the real enemy data that the player has interacted with.

  * No need to add new content; the model can use existing content to optimize your games.

  * More risky than the non-diverse models, but gives more variation in enemy data generation.

  * Minimal implementation takes a minimum of 1 hour using DataPredict™.

* [Creating Conditional-Diversity-Based Enemy Data Generation Model](HighValueProjectTutorials/CreatingConditionalDiversityBasedEnemyDataGenerationModel.md)

  * Uses players' combat data for generator's input and enemies' data as generator's output. The generated enemy data is then used against the discriminator to see if the player will interact with it.
 
    * Think of the discriminator as "fake player" and the discriminator will see the real enemy data that the player has interacted with.

  * No need to add new content; the model can use existing content to optimize your games.

  * More risky than the non-diverse models, but gives more variation in enemy data generation.

  * Minimal implementation takes a minimum of 2 hour using DataPredict™.

* [Creating Reward-Maximization-Based Difficulty Generation Model](HighValueProjectTutorials/CreatingRewardMaximizationBasedDifficultyGenerationModel.md)

  * Every time an enemy is killed, the positive reward tells the model to "make more enemies similar to this". 

  * If the player ignores or doesn't kill the enemy, the negative reward tells the model that "this enemy is not interesting to the player" or "this enemy is too hard for the player to kill".

  * No need to add new content; the model can use existing content to optimize your games.

  * Have higher play time potential due to its ability to exploit and explore than the other two models, but tend to be risky to use.

  * Minimal implementation takes a minimum of 2 hours using DataPredict™.

## Targeting Systems

* [Creating Distance-Minimization-Based Clustering Targeting Model](HighValueProjectTutorials/CreatingDistanceMinimizationBasedClusteringTargetingModel.md)

  * Find the center of players based on number of clusters.

  * Best suited for precise targeting to multiple areas.

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

* [Creating Probability-Maximization-Based Clustering Targeting Model](HighValueProjectTutorials/CreatingProbabilityMaximizationBasedClusteringTargetingModel.md)

  * Produces clusters that maximizes the likelihood of being hit.

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

* [Creating Probability-Based Targeting Model](HighValueProjectTutorials/CreatingProbabilityBasedTargetingModel.md)

  * Find the center of players by finding the area with high player density.

  * Can perform both precise and intentionally inaccurate, yet likely-to-hit targeting.

  * Only one cluster.

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

* [Creating Reward-Maximization-Based Targeting Model](HighValueProjectTutorials/CreatingRewardMaximizationBasedTargetingModel.md)

  * If your map has terrains and structures, the model may learn to corner and "trap" players to certain locations for easy "kills" or "targets".

  * Limited to one target at a time, but can take in multiple player locations.

  * Might be the most terrible idea out of this list. However, I will not stop game designers from making their games look "smart".

    * The model will likely do a lot of exploration before it can hit a single player. Once that particular location is marked as "reward location", the model will might overfocus on it.

  * Minimal implementation takes a minimum of 2 hours using DataPredict™.

## AI Players

* Creating Data-Based AI Players

  * Uses real players' data so that the AI players mimic real players.
 
  * Matches with real players' general performance.

* Creating Reward-Maximization-Based AI Players

  * Allows the creation of AI players that maximizes positive rewards.
 
  * May outcompete real players.

  * May exploit bugs and glitches.

* Creating Data-Based Reactionary AI Players

  * Same as data-based AI players.
 
  * The only difference is that you give counter attacks to players' potential attacks.

  * Best for mixing machine learning with game designers' control.

* Creating Reward-Maximization-Based Reactionary AI Players

  * Same as reward-maximization-based AI players.
 
  * The only difference is that you give counter attacks to players' potential attacks.

  * Best for mixing reinforcement learning with game designers' control.

  * Breaks mathematical theoretical guarantees due to inteference from game designers' control instead of model's own actions. Therefore, it is risky to use.

## Quality Assurance

* Creating Reward-Maximization-Based AI Bug Hunter

  * For a given "normal" data, the AI is rewarded based on how far the difference of the collected data compared to current data.

* Creating Curiosity-Based AI Bug Hunter

  * The AI will maximize actions that allows it to explore the game.

## Priority Systems

* Creating Probability-Based Priority System

* Creating Regression-Based Priority System

## Anti-Cheat Systems

* [Creating Anomaly Detection Model](HighValueProjectTutorials/CreatingAnomalyDetectionModel.md)

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

## Placement Systems

* [Creating Probability-Based Clustering Placement Model](HighValueProjectTutorials/CreatingProbabilityBasedClusteringPlacementModel.md)

  * Identifies areas of low player/item densities for spawning players/items.

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.
