# Creating Simple Data-Based Reactionary AI Player Model

For this tutorial, we need Markov model to build player state prediction model.

## Designing Our Feature Vector And Classes List

Before we start creating our model, we first need to visualize on how we will design our data to that our model can perform player state prediction.

### StatesList

```lua

local PlayerStatesList = {

    "PlayerAttack",
    "PlayerBlock",
    "PlayerFollow",
    "PlayerEscaping",
    "PlayerPickingUpItem",

}

```
## Constructing Our Model

```lua

local PlayerStatePredictionModel = DataPredict.Model.TabularQLearning.new({StatesList = StatesList})

```

## Training

```lua

-- Here, you notice that there is a reward value being inserted here. Generally, when you first call this, the reward value should be zero.

PlayerStatePredictionModel:train(previousPlayerStateVector, currentPlayerStateVector)

```

## Counter-Attacking

In order to assign the reward to that event is selected, we must first deploy the chosen event and observe if the player stayed for that event.

Below, it shows an example code for this.

```lua

local counterFunctionDictionary = {

  ["PlayerAttack"] = enemyBlock,
  ["PlayerBlock"] = enemyGoBehind,
  ["PlayerFollow"] = enemyAttack,
  ["PlayerEscaping"] = enemyFollow,
  ["PlayerPickingUpItem"] = enemyAttack,

}

local function run(Player)

    local isPlayerInServer = true

    local currentPlayerState

    local nextPlayerState

    local previousPlayerState

    local counterFunction

    while isPlayerInServer do

        currentPlayerState = getPlayerState(Player)

        PlayerStatePredictionModel:train(previousPlayerState, currentPlayerState)
    
        nextPlayerState = PlayerStatePredictionModel:predict(currentPlayerState)

        counterFunction = counterFunctionDictionary[nextPlayerState]

        if (counterFunction) then counterFunction() end

        isPlayerInServer = checkIfPlayerIsInServer(Player)

        previousPlayerState = currentPlayerState

        task.wait()

    end

end

```

## Model Parameters Loading 

In here, we will use our model parameters so that it can be used to load out models. There are three cases in here:

1. The player is a first-time player.

2. The player is a returning player.

3. Every player uses the same global model.

### Case 1: The Player Is A First-Time Player

Under this case, this is a new player that plays the game for the first time. In this case, we do not know how this player would act.

We have a multiple way to handle this issue:

* We create a "global" model that trains from every player, and then make a deep copy of the model parameters and load it into our models.

* We take from other players' existing model parameters and load it into our models.

### Case 2: The Player Is A Returning Player

Under this case, you can continue using the existing model parameters that was saved in Roblox's Datastores.

```lua

--[[

ModelParameters = PlayerStatePredictionModel:getModelParameters()

PlayerStatePredictionModel:setModelParameters(ModelParameters)

```

### Case 3: Every Player Uses The Same Global Model

Under this case, the procedure is the same to case 2 except that you need to:

* Load model parameters upon server start.

* Perform auto-save with the optional ability of merging with saved model parameters from other servers.

That's all for today!
