# DataPredict™

![DataPredict Icon](icons/DataPredictIcon.png)

--------------------------------------------------------------------

## THIS IS A SOURCE AVAILABLE CODE! NOT OPEN SOURCE! 

--------------------------------------------------------------------

| Version | Current Version Number |
|---------|------------------------|
| Release | 2.37                   |
| Beta    | 2.37.0                 |

--------------------------------------------------------------------

Aqwam's Machine, Deep And Reinforcement Learning Library (Aqwam-MDRLL)

Author: Aqwam Harish Aiman
	
Email: aqwam.harish.aiman@gmail.com

LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
--------------------------------------------------------------------

View the documentation here: https://aqwamcreates.github.io/DataPredict/

By using or possessing any copies of this library or its assets (including the icons), you agree to our [Terms And Conditions](docs/TermsAndConditions.md).

For information regarding potential license violations and eligibility for a bounty reward, please refer to the [Terms And Conditions Violation Bounty Reward Information](docs/TermsAndConditionsViolationBountyRewardInformation.md).

--------------------------------------------------------------------

Number of algorithms per model type:

| Model Type                     | Purpose                                         | Count |
|--------------------------------|-------------------------------------------------|-------|
| Regression                     | Continuous Value Prediction                     | 13    |
| Classification                 | Feature-Class Prediction                        | 13    |
| Clustering                     | Feature Grouping                                | 10    |
| Deep Reinforcement Learning    | State-Action Optimization Using Neural Networks | 26    |
| Tabular Reinforcement Learning | State-Action Optimization Using Tables          | 17    |
| Sequence Modelling             | Next State Prediction And Generation            | 3     |
| Filtering                      | Next State Tracking / Estimation                | 4     |
| Outlier Detection              | Outlier Score Generation                        | 4     |
| Recommendation                 | User-Item Pairing                               | 5     |
| Generative                     | Feature To Novel Value                          | 4     |
| Feature-Class Containers       | Feature-Class Look Up                           | 1     |
| Total                          |                                                 | 100   |

--------------------------------------------------------------------

* For strong deep learning applications, have a look at [DataPredict™ Neural](https://aqwamcreates.github.io/DataPredict-Neural) (object-oriented, static graph) and [DataPredict™ Axon](https://aqwamcreates.github.io/DataPredict-Axon) (function-oriented, dynamic graph) instead. DataPredict™ is only suitable for general purpose machine, deep and reinforcement learning.

  * Uses reverse-mode automatic differentiation and lazy differentiation evaluation.

  * Includes convolutional, pooling, embedding, dropout and activation layers.

  * Contains most of the deep reinforcement learning and generative algorithms listed here.

* Currently, DataPredict™ has ~93% (92 out of 99) models with online learning capabilities. By default, most models would perform offline / batch training on the first train before switching to online / incremental / sequential after the first train.

* Tabular reinforcement learning models can use optimizers. And yes, I am quite aware that I have overengineered this, but I really want to make this a grand finale before I stop updating DataPredict™ for a long time.

* No dimensionality reduction algorithms due to not being suitable for game-related use cases. They tend to be computationally expensive and are only useful when a full dataset is collected. This can be offset by choosing proper features and remove the unnecessary ones.

* No tree models (like decision trees) for now due to these models requiring the full dataset and tend to be computationally expensive. In addition, most of these tree models do not have online learning capabilities.

* Going "Independence" on my birthday at 23 January 2026. Probably.
