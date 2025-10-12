# Roadmap

## Core

The list of items shown below are likely to be implemented due to their mainstream use, ability to increase learning speed, or ability to reduce computational resources.

* Online Decision Trees And Boosting Algorithms

  * Currently, the offline variants offer superior performance in terms of generalization for tabular datasets. However, because they tend to be computationally expensive and requires a full dataset, it is not suitable for real-time game environments.

  * The research literature on the online variants of these algorithms are lacking, and so we are waiting for more papers to come out.

  * Additionally, we lack experience in developing decision trees and boosting algorithms, which may result in long development times of these algorithms.

* Incremental DBSCAN

  * It is an online version of DBSCAN that allows it to construct clusters from individual datapoints.

* Double Version Of Tabular Reinforcement Learning Algorithms.

  * This includes:
  
    * TabularDoubleQLearningV1
    
    * TabularDoubleQLearningV2
      
    * TabularClippedDoubleQLearning
  
    * TabularDoubleStateActionRewardStateActionV1
   
    * TabularDoubleStateActionRewardStateActionV2
    
    * TabularDoubleExpectedStateActionRewardStateActionV1
    
    * TabularDoubleExpectedStateActionRewardStateActionV2

  * My head doesn't want to let this go.

## Nice-To-Have

The list of items shown below may not necessarily be implemented in the future. However, they could be prioritized with external demand, collaboration, or funding.

* None
