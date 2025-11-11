# Creating Next Zone Prediction Model

For this tutorial, we need DynamicBayesianNetwork model to build player state prediction model.

## Designing Our Zones List

Before we start creating our model, we first need to visualize on how we will design our data to that our model can perform next zone predictions.

### StatesList

```lua

local ZonesList = {

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

## Training

```lua

NextZonePredictionModel:train(previousPlayerStateVector, currentPlayerStateVector)

```

## Executing Predicted Player States

In order to assign the reward to that event is selected, we must first deploy the chosen event and observe if the player stayed for that event.

Below, it shows an example code for this.

```lua

local function run(Player)

    local isPlayerInServer = true

    local currentPlayerState

    local nextPlayerState

    local previousPlayerState

    local playerFunction

    while isPlayerInServer do

        currentPlayerState = getPlayerState(Player)

        PlayerStatePredictionModel:train(previousPlayerState, currentPlayerState)
    
        nextPlayerState = PlayerStatePredictionModel:predict(currentPlayerState)

        playerFunction = playerFunctionDictionary[nextPlayerState]

        if (playerFunction) then playerFunction() end

        isPlayerInServer = checkIfPlayerIsInServer(Player)

        previousPlayerState = currentPlayerState

        task.wait()

    end

end

```
