# Low-Risk Systems

## Retention Systems

The worse case scenario is that the model either predicts too early or too late for appropriate intervention. The former leads to flooding a lot of events that could lead to player getting engaged, while the latter scenario is and equivalent of you not having a model to intervene and the player will leave anyways.

* [Creating Low-Risk Time-To-Leave Prediction Model](RetentionSystems/CreatingLowRiskTimeToLeavePredictionModel.md)

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

## Targeting Systems

The worse case scenario is that the model will just miss the players, which is a desirable property for games since they are more focused in making the AIs fun and playable instead of accurate and difficult.

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

## Load Shedding

The worse case scenario is that the model will place the asset that are far away from the players, but this isn't a problem given that most games literally place assets at random places for players to collect.

* [Creating Probability-Based Clustering Placement Model](LoadSheddingSystems/CreatingProbabilityBasedClusteringPlacementModel.md)

  * Identifies areas of high player densities for spawning assets.
 
  * Reduces the servers' and clients' computational resources by spawning items in places that are highly likely to be interact by the players.

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.
