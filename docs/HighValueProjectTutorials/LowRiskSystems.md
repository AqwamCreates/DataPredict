# Low Risk Systems

# Retention Systems

The worse case is that the model either predicts too early or too late for appropriate intervention. The former leads to flooding a lot of events that could lead to player getting engaged, while the latter scenario is and equivalent of you not having a model to intervene and the player will leave anyways.

* [Creating Low-Risk Time-To-Leave Prediction Model](RetentionSystems/CreatingLowRiskTimeToLeavePredictionModel.md)

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

## Targeting Systems

The worse case the the model will just miss the players, which is a desirable property for games since they are more focused in making the AI fun and escapable instead of accurate and difficult.

* [Creating Distance-Minimization-Based Clustering Targeting Model](TargetingSystems/CreatingDistanceMinimizationBasedClusteringTargetingModel.md)

  * Find the center of players based on number of clusters.

  * Best suited for precise targeting to multiple areas.

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

* [Creating Probability-Maximization-Based Clustering Targeting Model](TargetingSystems/CreatingProbabilityMaximizationBasedClusteringTargetingModel.md)

  * Produces clusters that maximizes the likelihood of being hit.

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

* [Creating Probability-Based Targeting Model](TargetingSystems/CreatingProbabilityBasedTargetingModel.md)

  * Find the center of players by finding the area with high player density.

  * Can perform both precise and intentionally inaccurate, yet likely-to-hit targeting.

  * Only one cluster.

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.
