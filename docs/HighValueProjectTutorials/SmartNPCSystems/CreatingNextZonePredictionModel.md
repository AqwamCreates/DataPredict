# Creating Next Zone Prediction Model

For this tutorial, we need DynamicBayesianNetwork model to build player state prediction model.

## Designing Our Zones List

Before we start creating our model, we first need to visualize on how we will design our data to that our model can perform next zone predictions.

### StatesList

```lua

local zonPl = {

    "Mall",
    "Bank",
    "RetailStore",
    "PetrolStation",

}

```
## Constructing Our Model

```lua

local NextZonePredictionModel = DataPredict.Model.DynamicBayesianNetwork.new()

```

## Main Code

```lua

local previousZonePlayerCountVector

local currentZonePlayerCountVector

local nextZoneProbabilityVector

local function onZoneEnter()

    currentZonePlayerCountVector = getZonePlayerCountVector()

    NextZonePredictionModel:train(previousZonePlayerCountVector, currentZonePlayerCountVector)

    previousZonePlayerCountVector = currentZonePlayerCountVector

    nextZoneProbabilityVector = NextZonePredictionModel:predict(currentZonePlayerCountVector)

    assignGuards(nextZoneProbabilityVector)

end

```
