# Game Design Meets Machine Learning

Note: This documentation is still under construction. There will be links that go more in depth.

* Measurement Of Fun

  * Session Length -> The more the player is engaged, the longer the player stays.

  * Map Coverage -> The more the player is engaged, the more the player explores.

  * Variety Of Items Collected -> The more the player is engaged, the more player collect different items.

* Intepreting Local And Global Optima In Game Design

  * Local Optima -> The best solution for anyting related to the current game session.
 
  * Global Optima -> The best solution for all game sessions.

* What's Your Goal?

  * Reward Maximization -> Use "measurement of fun" metric as rewards and combine it with reinforcement learning models.
 
  * Prediction -> Use regression and classification models.
 
  * Best Middle Values -> Use clustering models.

* Game Environment Data Is Far More Cleaner Than Real World Data

  * Noise usually comes from overlapping interactions.
 
  * Your Global Optimum might be a real Global Optimum.
 
  * Game environment states are just a series of physics calculations. Your model may accidentally associate certain things with certain states!

* Model Calculation Speed Vs The Game Engine

  * Per Frame (Physics/Render) -> Model must be fast. Ideally use single datapoints or online models here.

  * Per Interval -> Model calculation time must not exceed the interval. Ideally use mini-batch data here.
  
