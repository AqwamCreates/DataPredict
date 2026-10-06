# Creating Probability-Based Clustering Placement Model

Hi guys! In this tutorial, we will demonstrate on how to create cluster-based placement model.

You must use Expectation-Maximization model.

## Initializing The Clustering Model

```lua

-- For this tutorial, we will let the model decide how many clusters it will produce based on player / asset spread.

-- Note, we're setting math.huge here, but that doesn't mean we will begin producing an infinite amount of clusters! It will start at 1 and increases it until the model finds a suitable number of clusters.

local PlacementModel = DataPredict.Models.ExpectationMaximization.new({numberOfClusters = math.huge})

```

## Collecting The Players' Locations

In order to find the center of the clusters, we first need all the players' location data and put them into a matrix.

```lua

local playerLocationDataMatrix = {

  {player1LocationX, player1LocationY, player1LocationZ},
  {player2LocationX, player2LocationY, player2LocationZ},
  {player3LocationX, player3LocationY, player3LocationZ},
  {player4LocationX, player4LocationY, player4LocationZ},
  {player5LocationX, player5LocationY, player5LocationZ},
  {player6LocationX, player6LocationY, player6LocationZ},
  {player7LocationX, player7LocationY, player7LocationZ},

}

```

## Getting The Center Of Clusters

Once you collected the players' location data, you must call model's train() function. This will generate the center of clusters to the model parameters.

```lua

PlacementModel:train(playerLocationDataMatrix)

```

Once train() is called, call the getModelParameters() function to get the center of cluster location data.

```lua

local centroidMatrix = PlacementModel:getModelParameters()

centroidMatrix = centroidMatrix[1]

```

## Interacting With The Center Of Clusters

Since we have dynamic number of clusters, we can expect multiple rows for our matrix. As such we can process our game logic here.

### Asset Placement

```lua

local function placeAssetAtRandomLocation(Asset)

  local ModelParameters = PlacementModel:getModelParameters()

  local meanMatrix = ModelParameters[1]
  
  local varianceMatrix = ModelParameters[2]
  
  local numberOfClusters = #meanMatrix
  
  local randomClusterIndex = math.random(1, numberOfClusters)

  local randomUnwrappedMeanVector = meanMatrix[randomClusterIndex]

  local randomUnwrappedVarianceVector = varianceMatrix[randomClusterIndex]

  local x = randomUnwrappedMeanVector[1] + ((math.random() * 2 - 1) * randomUnwrappedVarianceVector[1])

  local y = randomUnwrappedMeanVector[2] + ((math.random() * 2 - 1) * randomUnwrappedVarianceVector[2])

  local z = randomUnwrappedMeanVector[3] + ((math.random() * 2 - 1) * randomUnwrappedVarianceVector[3])

  placeAsset(asset, x, y, z)

end

```

### Asset Removal

Asset removal can be handled using multiple strategies. Instead of forcing developers into one fixed rule, you can provide several removal methods and let them choose the behavior that best fits their game.

The methods below range from more model-aware removal techniques to simpler fallback methods.

> You only need to use one of these removal methods at a time.

---

#### Shared Helpers

These helper functions are used by several removal methods below.

```lua
local AssetFolder = workspace.AssetFolder

-- Prevents extremely small variances from producing unstable scores.
local MIN_VARIANCE = 1e-6

-- Optional: cluster weights.
-- If nil, all clusters are treated equally.
-- Example:
-- local clusterWeightVector = {0.55, 0.30, 0.15}
local clusterWeightVector = nil

local function getAssetPosition(Asset)
    if Asset:IsA("Model") then
        return Asset:GetPivot().Position
    elseif Asset:IsA("BasePart") then
        return Asset.Position
    end

    return nil
end

local function getLogDiagonalGaussian(positionValues, meanVector, varianceVector)
    local logDensity = 0

    for i = 1, 3 do
        local variance = math.max(varianceVector[i], MIN_VARIANCE)
        local delta = positionValues[i] - meanVector[i]

        logDensity = logDensity + (-0.5 * (((delta * delta) / variance) + math.log(2 * math.pi * variance)))
    end

    return logDensity
end

local function getMixtureLogLikelihood(position, meanMatrix, varianceMatrix)
    local clusterCount = #meanMatrix

    if clusterCount == 0 then
        return -math.huge
    end

    local positionValues = {position.X, position.Y, position.Z}

    local totalWeight = 0

    if clusterWeightVector then
        for i = 1, clusterCount do
            totalWeight = totalWeight + (clusterWeightVector[i] or 0)
        end
    end

    if totalWeight <= 0 then
        totalWeight = clusterCount
    end

    local logComponents = {}
    local maxLog = -math.huge

    for clusterIndex = 1, clusterCount do
        local weight = 1 / clusterCount

        if clusterWeightVector then
            weight = (clusterWeightVector[clusterIndex] or 0) / totalWeight
        end

        weight = math.max(weight, 1e-12)

        local logWeight = math.log(weight)
        local logDensity = getLogDiagonalGaussian(
            positionValues,
            meanMatrix[clusterIndex],
            varianceMatrix[clusterIndex]
        )

        local logComponent = logWeight + logDensity

        table.insert(logComponents, logComponent)

        if logComponent > maxLog then
            maxLog = logComponent
        end
    end

    local sumExp = 0

    for _, logComponent in ipairs(logComponents) do
        sumExp = sumExp + math.exp(logComponent - maxLog)
    end

    return maxLog + math.log(sumExp)
end
```

---

#### Likelihood-Based Removal

This method removes assets that have low likelihood under the trained placement model.

Instead of checking whether an asset is close to a cluster center, this checks whether the asset position is plausible according to the full cluster distribution.

This is the recommended replacement for hard distance removal.

```lua
local function removeAllAssetsWithLowLikelihood(logLikelihoodThreshold)
    local ModelParameters = PlacementModel:getModelParameters()

    if not ModelParameters then
        return
    end

    local meanMatrix = ModelParameters[1]
    local varianceMatrix = ModelParameters[2]

    if not meanMatrix or not varianceMatrix then
        return
    end

    for _, Asset in ipairs(AssetFolder:GetChildren()) do
        local assetPosition = getAssetPosition(Asset)

        if assetPosition then
            local logLikelihood = getMixtureLogLikelihood(
                assetPosition,
                meanMatrix,
                varianceMatrix
            )

            if logLikelihood < logLikelihoodThreshold then
                Asset:Destroy()
            end
        end
    end
end
```

Example usage:

```lua
-- The correct threshold depends on your map scale, variance values, and desired strictness.
-- Lower log-likelihood means the asset is less likely under the model.
removeAllAssetsWithLowLikelihood(-25)
```

This method works well when you want removal to respect cluster spread, variance, and overall player-location density.

---

#### Percentile-Based Likelihood Removal

If you do not want to manually tune an absolute likelihood threshold, you can remove the lowest-scoring percentage of assets.

This is useful when the raw log-likelihood values are difficult to interpret.

```lua
local function removeBottomLikelihoodPercentile(percentToRemove)
    local ModelParameters = PlacementModel:getModelParameters()

    if not ModelParameters then
        return
    end

    local meanMatrix = ModelParameters[1]
    local varianceMatrix = ModelParameters[2]

    if not meanMatrix or not varianceMatrix then
        return
    end

    local scoredAssets = {}

    for _, Asset in ipairs(AssetFolder:GetChildren()) do
        local assetPosition = getAssetPosition(Asset)

        if assetPosition then
            local logLikelihood = getMixtureLogLikelihood(
                assetPosition,
                meanMatrix,
                varianceMatrix
            )

            table.insert(scoredAssets, {
                instance = Asset,
                score = logLikelihood,
            })
        end
    end

    table.sort(scoredAssets, function(a, b)
        return a.score < b.score
    end)

    local removeCount = math.floor(#scoredAssets * math.clamp(percentToRemove, 0, 1))

    for i = 1, removeCount do
        scoredAssets[i].instance:Destroy()
    end
end
```

Example usage:

```lua
-- Removes the lowest-scoring 30% of assets.
removeBottomLikelihoodPercentile(0.3)
```

This method is easier to control when you want relative culling rather than an absolute model threshold.

---

#### Mahalanobis Distance Removal

This method is a variance-aware distance method.

Instead of using raw Euclidean distance, it normalizes distance by the cluster variance. This makes it more flexible than simple distance checking.

It is still a hard-threshold method, but the threshold is based on cluster spread rather than world-space distance alone.

```lua
local function getMinimumMahalanobisDistanceSquared(position, meanMatrix, varianceMatrix)
    local positionValues = {position.X, position.Y, position.Z}
    local bestDistanceSquared = math.huge

    for clusterIndex, meanVector in ipairs(meanMatrix) do
        local varianceVector = varianceMatrix[clusterIndex]

        if varianceVector then
            local distanceSquared = 0

            for i = 1, 3 do
                local variance = math.max(varianceVector[i], MIN_VARIANCE)
                local delta = positionValues[i] - meanVector[i]

                distanceSquared = distanceSquared + ((delta * delta) / variance)
            end

            if distanceSquared < bestDistanceSquared then
                bestDistanceSquared = distanceSquared
            end
        end
    end

    return bestDistanceSquared
end

local function removeAllAssetsOutsideMahalanobisThreshold(thresholdSquared)
    local ModelParameters = PlacementModel:getModelParameters()

    if not ModelParameters then
        return
    end

    local meanMatrix = ModelParameters[1]
    local varianceMatrix = ModelParameters[2]

    if not meanMatrix or not varianceMatrix then
        return
    end

    for _, Asset in ipairs(AssetFolder:GetChildren()) do
        local assetPosition = getAssetPosition(Asset)

        if assetPosition then
            local distanceSquared = getMinimumMahalanobisDistanceSquared(
                assetPosition,
                meanMatrix,
                varianceMatrix
            )

            if distanceSquared > thresholdSquared then
                Asset:Destroy()
            end
        end
    end
end
```

Example usage:

```lua
-- Approximate 3D Gaussian confidence thresholds:
-- 90%: 6.251
-- 95%: 7.815
-- 99%: 11.345
removeAllAssetsOutsideMahalanobisThreshold(7.815)
```

This method is useful if you want something simpler than full likelihood scoring but still more intelligent than plain distance.

---

#### Stochastic Removal

This method removes assets based on probability derived from their likelihood score.

Assets with lower likelihood have a higher chance of being removed, while assets with higher likelihood are more likely to remain.

This is useful for softer, less binary removal behavior.

```lua
local function removeAssetsByLikelihoodProbability(removalStrength)
    removalStrength = math.clamp(removalStrength, 0, 1)

    local ModelParameters = PlacementModel:getModelParameters()

    if not ModelParameters then
        return
    end

    local meanMatrix = ModelParameters[1]
    local varianceMatrix = ModelParameters[2]

    if not meanMatrix or not varianceMatrix then
        return
    end

    local scoredAssets = {}
    local minScore = math.huge
    local maxScore = -math.huge

    for _, Asset in ipairs(AssetFolder:GetChildren()) do
        local assetPosition = getAssetPosition(Asset)

        if assetPosition then
            local logLikelihood = getMixtureLogLikelihood(
                assetPosition,
                meanMatrix,
                varianceMatrix
            )

            table.insert(scoredAssets, {
                instance = Asset,
                score = logLikelihood,
            })

            if logLikelihood < minScore then
                minScore = logLikelihood
            end

            if logLikelihood > maxScore then
                maxScore = logLikelihood
            end
        end
    end

    local scoreRange = maxScore - minScore

    for _, entry in ipairs(scoredAssets) do
        local normalizedScore = 1

        if scoreRange > 1e-9 then
            normalizedScore = (entry.score - minScore) / scoreRange
        end

        local removeProbability = (1 - normalizedScore) * removalStrength

        if math.random() < removeProbability then
            entry.instance:Destroy()
        end
    end
end
```

Example usage:

```lua
-- Low-likelihood assets have up to a 75% chance of being removed.
removeAssetsByLikelihoodProbability(0.75)
```

This method is less deterministic, so it is best used for decorative or non-critical assets.

---

#### Distance-Based Removal

This is the simplest removal method. It removes assets that are farther than a fixed distance from every cluster center.

This method is easy to understand and tune, but it does not account for cluster variance or cluster shape.

```lua
local function removeAllAssetsWithDistanceThreshold(distanceThreshold)
    local ModelParameters = PlacementModel:getModelParameters()

    if not ModelParameters then
        return
    end

    local meanMatrix = ModelParameters[1]

    if not meanMatrix then
        return
    end

    for _, Asset in ipairs(AssetFolder:GetChildren()) do
        local assetPosition = getAssetPosition(Asset)

        if assetPosition then
            local keepAsset = false

            for _, meanVector in ipairs(meanMatrix) do
                local distanceVector = Vector3.new(
                    assetPosition.X - meanVector[1],
                    assetPosition.Y - meanVector[2],
                    assetPosition.Z - meanVector[3]
                )

                if distanceVector.Magnitude <= distanceThreshold then
                    keepAsset = true
                    break
                end
            end

            if not keepAsset then
                Asset:Destroy()
            end
        end
    end
end
```

Example usage:

```lua
removeAllAssetsWithDistanceThreshold(10)
```

This method is best used when you want a simple, predictable radius-based removal rule.

## Resetting Our Placement System

By default, when you reuse the machine learning models from DataPredict, it will interact with the existing model parameters. As such, we need to reset the model parameters by calling the setModelParameters() function and set it to "nil".

```lua

PlacementModel:setModelParameters(nil)

```

That's all for today!
