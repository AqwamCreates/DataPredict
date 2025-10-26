# Retention Systems

* [Creating Time-To-Leave Prediction Model](RetentionSystems/CreatingTimeToLeavePredictionModel.md)

  * No need to add new content; the model can use existing content to optimize your games.

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

* [Creating Probability-To-Leave Prediction Model](RetentionSystems/CreatingProbabilityToLeavePredictionModel.md)

  * No need to add new content; the model can use existing content to optimize your games.

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

* [Creating Probabilistic Time-To-Leave Prediction Model](RetentionSystems/CreatingProbabilisticTimeToLeavePredictionModel.md)

  * Combines both "Time-To-Leave Prediction Model" and "Probability-To-Leave Prediction Model".

  * No need to add new content; the model can use existing content to optimize your games.

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

* [Creating Probability-To-Interact Prediction Model](RetentionSystems/CreatingProbabilityToInteractPredictionModel.md)

  * Can be combined with generative and reward-maximization-based models for optimized retention and interaction.

  * No need to add new content; the model can use existing content to optimize your games.

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

* [Creating Left-Too-Early Detection Model](RetentionSystems/CreatingLeftTooEarlyDetectionModel.md)

   * Inverse of probability-to-leave model by detecting outliers.

   * No need to add new content; the model can use existing content to optimize your games.

   * Highly exploitable if the player accumulates long session times over many sessions before suddenly decrease the session times gradually if rewards are involved.

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

* [Creating Labelless Left-Too-Early Detection Model](RetentionSystems/CreatingLabellessLeftTooEarlyDetectionModel.md)

  * Same as "Left-Too-Early Detection Model", but it does not require manual tracking of label data, which makes it less accurate.

  * Minimal implementation takes a minimum of 30 minutes using DataPredict™.

* [Creating Deep Play Time Maximization Model](RetentionSystems/CreatingDeepPlayTimeMaximizationModel.md)

  * The model chooses actions or events that maximizes play time.

  * No need to add new content; the model can use existing content to optimize your games.

  * Have higher play time potential due to its ability to exploit and explore than the other four models, but tend to be risky to use.

  * Minimal implementation takes a minimum of 2 hours using DataPredict™, especially if custom events are associated with the model's output.

* [Creating Simple Play Time Maximization Model](RetentionSystems/CreatingSimplePlayTimeMaximizationModel.md)

  * Uses discrete input values (e.g. "focus", "run" and "attack") to maximize play time.

  * No need to add new content; the model can use existing content to optimize your games.

  * More safer and faster to learn but more limited in expressive power compared to the deep version.

  * Minimal implementation takes a minimum of 1 hour using DataPredict™, especially if custom events are associated with the model's output.

* [Creating Junior-Senior Play Time Maximization Ensemble Model](RetentionSystems/CreatingJuniorSeniorPlayTimeMaximizationEnsembleModel.md)

  * Uses a combination of:
 
    * Simple Play Time Maximization Model
   
    * Deep Play Time Maximization Model

  * When the simple model chooses to "consult" the deep model, the deep model will generate actions instead of the simple model.

  * Less risky and learns faster than the original "Deep Play Time Maximization Model", but takes more time to implement.

  * Minimal implementation takes a minimum of 3 hours using DataPredict™.

* [Creating Gated Deep Play Time Maximization Ensemble Model](RetentionSystems/CreatingGatedDeepPlayTimeMaximizationEnsembleModel.md)

  * Uses a combination of:
 
    * Time-To-Leave Prediction Model
   
    * Probability-To-Leave Prediction Model
   
    * Deep Play Time Maximization Model

  * Less risky than the original "Deep Play Time Maximization Model", but takes more time to implement.

  * Minimal implementation takes a minimum of 4 hours using DataPredict™.

* [Creating Gated Junior-Senior Play Time Maximization Ensemble Model](RetentionSystems/CreatingGatedJuniorSeniorPlayTimeMaximizationEnsembleModel.md)

  * Uses a combination of:
 
    * Time-To-Leave Prediction Model
   
    * Probability-To-Leave Prediction Model

    * Simple Play Time Maximization Model

    * Deep Play Time Maximization Model

  * The least riskiest model out there for play time maximization, but takes the longest time to implement.

  * Minimal implementation takes a minimum of 6 hours using DataPredict™.

* [Creating Engagement Milestone Detection Ensemble Model](RetentionSystems/CreatingEngagementMilestoneDetectionEnsembleModel.md)

  * Uses a combination of:
 
    * Time-To-Leave Prediction Model
   
    * Probability-To-Leave Prediction Model
   
    * Left-Too-Early Detection Model

  * The model periodically checks if the player is playing much more longer or more engaged than usual.

  * Minimal implementation takes a minimum of 4 hours using DataPredict™.
