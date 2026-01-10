# [Economy Systems](../EconomySystems.md) - Creating Base Price Search Model

Hi guys! In this tutorial, we will demonstrate on how to create cluster-based search algorithm to find the base prices. Below, I will show you the valid algorithms that you can use for these models and its properties.

| Model                      | Maximum Cluster Count | Objective                                                                                                                       | Objective (In Terms Of Emotional Perspective)                               |
|----------------------------|-----------------------|---------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| K-Means                    | Infinity.             | Gives hard assigments of players' spending behaviour and find the optimal base price for that particular item.                  | Strict Base Price.                                                          |
| Fuzzy CMeans               | Infinity.             | Gives overlapping assignments of players' spending behaviour and find the best overlapping base price for that particular item. | Negotiable Base Price.                                                      |
| ExpectationMaximization    | Infinity.             | Find the most likely base price that makes the players want to purchase that particular item.                                   | Players'  Desired Base Price.                                               |
| Mean Shift                 | Only 1.               | Finds the base price where a lot of players tend to make a purchase for that particular item.                                   | Popular Base Price.                                                         |
| Agglomerative Hierarchical | Infinity.             | Finds the base price by ranking the purchases.                                                                                  | Status-Based Base Price (e.g. "Subsidized Pricing For Low Income Earners"). |

## Initializing The Clustering Model

Before we can produce ourselves a search model, we first need to construct a model, which is shown below. Ensure that the distance function is not "CosineDistance".

We also recommend that for each item has their own model instead of combining multiple items to one model.

```lua

 -- For this tutorial, we will assume that we have three types of spenders: casual, strategist and whale. Hence 3 clusters.

local BasePriceSearchModel = DataPredict.Models.KMeans.new({numberOfClusters = 3, distanceFunction = "Euclidean"})

```

## Collecting The Players' Currencies

In here, we're assuming that we're getting the base price based on all players. 

If you want a player-specific pricing, then only include that player. However, I do not recommend this approach as it would lead to sparse data issue as not many purchases can be made by a single player.

```lua

local itemPaidUsingTheseCurrenciesDataMatrix = {

  {player1CashAmount, player1ManaResiduesAmount, player1GoldBarsAmount},
  {player2CashAmount, player2ManaResiduesAmount, player2GoldBarsAmount},
  {player3CashAmount, player3ManaResiduesAmount, player3GoldBarsAmount},
  {player4CashAmount, player4ManaResiduesAmount, player4GoldBarsAmount},
  {player5CashAmount, player5ManaResiduesAmount, player5GoldBarsAmount},
  {player6CashAmount, player6ManaResiduesAmount, player6GoldBarsAmount},
  {player7CashAmount, player7ManaResiduesAmount, player7GoldBarsAmount},

}

```

## Getting The Center Of Clusters

Once you collected the players' location data, you must call model's train() function. This will generate the center of clusters to the model parameters.

```lua

BasePriceSearchModel:train(itemPaidUsingTheseCurrenciesDataMatrix)

```

Once train() is called, call the getModelParameters() function to get the center of cluster location data.

```lua

local centroidMatrix = BasePriceSearchModel:getModelParameters()[1]

```

## Interacting With The Center Of Clusters

Since we have three clusters, we can expect three rows for our matrix. As such we can process our game logic here.

```lua

for clusterIndex, unwrappedClusterVector in ipairs(ModelParameters) do

  local baseCashAmount = unwrappedClusterVector[1]
  
  local baseManaResiduesAmount = unwrappedClusterVector[2]
  
  local baseGoldBarsAmount = unwrappedClusterVector[3]

  updateBasePriceForItem(item, baseCashAmount, baseManaResiduesAmount, baseGoldBarsAmount)

end

```

## Resetting Our Targeting System

By default, when you reuse the machine learning models from DataPredict, it will interact with the existing model parameters. As such, if you want a fresh start whenever you run the train() function, then call thesetModelParameters() function and set it to "nil".

```lua

BasePriceSearchModel:setModelParameters(nil)

```

## Determining The Clusters' Player Pricing Behaviour

Unfortunately, our models don't exactly tell which clusters belong to what player pricing behaviour. However, there are some workaround where you sum all the different types of currencies into one overall currency for each cluster. However, do not sum these total currencies with other clusters. Then, you can determine the players' pricing behaviour for each cluster where:

  * The one that has the highest overall currency belong to the "whale" group.
  
  * The one that has the lowest overall currency belong to the "casual" group.

## Expectation Maximization's Extra Ability (Optional Reading)

ExpectationMaximization actually includes the ability where you can increase or decrease based on how confident you are from the original base price. You can then use this to generate new base prices with a moving cluster for our algorithms, making it excellent for dynamic pricing.

The code below will show you on how to take advantage of this.

```lua

local ModelParameters = BasePriceSearchModel:getModelParameters()

local centroidMatrix = ModelParameters[1]

local varianceMatrix = ModelParameters[2]

local randomUnwrappedMeanVector = centroidMatrix[1] -- This index determines which cluster we want to access.

local randomUnwrappedVarianceVector = varianceMatrix[1] -- This index determines which cluster we want to access.

local modifiedBaseCashAmount = randomUnwrappedMeanVector[1] + ((math.random() * 2 - 1) * randomUnwrappedVarianceVector[1])

local modifiedBaseManaResiduesAmount = randomUnwrappedMeanVector[2] + ((math.random() * 2 - 1) * randomUnwrappedVarianceVector[2])

local modifiedBaseGoldAmount = randomUnwrappedMeanVector[3] + ((math.random() * 2 - 1) * randomUnwrappedVarianceVector[3])

```

That's all for today!
