# Game Design Meets Machine Learning Tutorials

Note: This documentation is still under construction. There will be links that go more in depth.

## Goal

* [Choosing The Correct Goal For The Model](GameDesignMeetsMachineLearningTutorials/ChoosingTheCorrectGoalForTheModel.md) (Incomplete)

## Engagement

* [Making Games Engaging Instead Of Accurate](GameDesignMeetsMachineLearningTutorials/MakingEngagingFunInsteadOfAccurate.md) (Incomplete)

* [Measurement Of Engagement](GameDesignMeetsMachineLearningTutorials/MeasurementOfEngagement.md)

* [Engagement-Based Reward Function Formula For Reinforcement Learning](GameDesignMeetsMachineLearningTutorials/EngagementBasedRewardFunctionFormulaForReinforcementLearning.md)

## Personalization

* [Personal VS Global Model Training](GameDesignMeetsMachineLearningTutorials/PersonalVSGlobalModelTraining.md) (Incomplete)

* [Session-Based Vs Cumulative-Based Model Training](GameDesignMeetsMachineLearningTutorials/SessionBasedVsCumulativeBasedModelTraining.md) (Incomplete)

## Performance

* [Game Frames VS Model Training](GameDesignMeetsMachineLearningTutorials/GameFramesVSModelTraining.md) (Incomplete)

* [Data Noise, Correlation And Causation](GameDesignMeetsMachineLearningTutorials/DataNoiseCorrelationAndCausation.md) (Incomplete)

## What's Your Goal?

* Goal Maximization -> Use "measurement of fun" metrics as rewards and combine it with reinforcement learning models.

* Prediction -> Use regression and classification models.

* Best Middle Values -> Use clustering models.

## Model Calculation Speed Vs The Game Engine

* Per Frame (Physics / Render) -> Model must be fast. Ideally use single datapoints or online models here.

* Per Interval -> Model calculation time must not exceed the interval. Ideally use mini-batch training here.

* Per Session End -> Batch training is allowed.

## Game Environment Data Is Far More Cleaner Than Real World Data

* Noise usually comes from overlapping interactions.
 
* Your model's global optimum might be a real global optimum.

* Game environment states are just a series of physics calculations. Your model may accidentally associate certain things with certain states!

## Intepreting Local And Global Optima In Game Design

* Local Optima -> The best solution for anything related to the current game session.
 
* Global Optima -> The best solution for all game sessions.
