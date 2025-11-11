# Creating Next Zone Prediction Model

For this tutorial, we need DynamicBayesianNetwork model to build player state prediction model.

## Designing Our Zone Player Count Vector

```lua

local zonePlayerCount = {

  {mallPlayerCount, bankPlayerCount, storePlayerCount, petrolStationPlayerCount},

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

    nextZoneProbabilityVector = NextZonePredictionModel:predict(currentZonePlayerCountVector)

    assignGuards(nextZoneProbabilityVector)

    previousZonePlayerCountVector = currentZonePlayerCountVector

end

```
