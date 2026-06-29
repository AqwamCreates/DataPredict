# [Smart Placement Systems](../SmartPlacementSystems.md) - Creating Object Placement Prediction Model

Hello guys! Today, I will be showing you on how to create a placement-based model that could predict the next object to place.

## Setting Up

```lua

local DataPredict = require(DataPredict)

--[[

  For incremental learning purposes, set the maximumNumberOfIterations to 1 to avoid fully train on the current data.

  Additionally, the more number of maximumNumberOfIterations you have, the lower the learningRate it should be to avoid "inf" and "nan" issues.

  Also please to make sure to set latentFactorCount to be less than the number of features to avoid overfitting.

--]]

local PlacementPredictionModel = DataPredict.Models.FactorizationMachine.new({maximumNumberOfIterations = 1, learningRate = 0.3, latentFactorCount = 5, binaryFunction = "Logistic"})

```

## Feature Matrix Format

```lua

local objectPlacementFeatureMatrix = {
    {
        1, -- We're just adding 1 here to add "bias".
        positionXPlacement, -- This is the previous object placement position made by the player.
        positionYPlacement,
        positionZPlacement,

        changeInPositionX, -- This is calculated based on previous object placement position and the target placement position. In other words, targetPositionX - positionXPlacement
        changeInPositionY,
        changeInPositionZ,

        objectRarityValue,
        objectCost,
        isInteractable, -- This only accepts 1 and 0. Additionally, you can also use 1 and -1.
        isAWallObject, -- Same as above.
        isALivingRoomObject, -- Same as above.
        isAKitchenObject, -- Same as above.
        isABathroomObject, -- Same as above.
        isABedroomObject, -- Same as above.
        isAGardenObject, -- Same as above.
        
        currentPlayerCashAmount, -- The mount of cash that the player is currently holding.
        consecutiveNumberOfTimesPlayerPlacedThisObject, -- The number of time that the player have placed this object in a row. Not to be confused with total number of this object placed by the player.
    }
}

```

## Upon Player Placement

By the time the player places an object, it is time for us to train the model. But first, we need to calculate the difference.

```lua

local objectPlacementFeatureMatrixToTrain = {}

local objectPlacementLabelVectorToTrain = {}

local function onPlacement(Player, ...) -- All the features from the previous feature matrix.

 -- In here, we fetch the object's attributes that was placed by the player.

 local objectPlacementUnwrappedFeatureVector = {

  1, -- We're just adding 1 here to add "bias".
  positionXPlacement, -- This is the previous object placement position made by the player.
  positionYPlacement,
  positionZPlacement,

  changeInPositionX, -- This is calculated based on previous object placement position and the target placement position.
  changeInPositionY,
  changeInPositionZ,

  objectRarityValue,
  objectCost,
  isInteractable,
  isAWallObject,
  isALivingRoomObject,
  isAKitchenObject,
  isABathroomObject,
  isABedroomObject,
  isAGardenObject,

  currentPlayerCashAmount,
  consecutiveNumberOfTimesPlayerPlacedThisObject,

}

 table.insert(objectPlacementFeatureMatrixToTrain, objectPlacementUnwrappedFeatureVector)

 table.insert(objectPlacementLabelVectorToTrain, {1}) -- This tells the model that the probability of placing this object is 100% for the given feature matrix.

 PlacementPredictionModel:train(objectPlacementFeatureMatrixToTrain, objectPlacementLabelVectorToTrain) -- Theoretically, you can use single data points per train() function call instead of accumulating data points. However, factorization machines works best in batches of data in order to handle sparsity better.

end

```

This should give you a model that predicts a rough estimate when they'll Placement.

Then, you must save the model parameters to Roblox's DataStores for future use.

```lua

local ModelParameters = PlacementPredictionModel:getModelParameters()

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

PlacementPredictionModel:setModelParameters(ModelParameters)

```

### Case 3: Every Player Uses The Same Global Model

Under this case, the procedure is the same to case 2 except that you need to:

* Load model parameters upon server start.

* Perform auto-save with the optional ability of merging with saved model parameters from other servers.

## Upon Player Placement Target

In order to produce recommendations from our model, we must perform this operation:

```lua

local function displayPredictedObject(Player, ...)

 -- In here, we fetch all the objects' attributes that can be placed by the player.

 local probabilityVector = PlacementPredictionModel:predict(objectPlacementFeatureMatrixToTrain, true) -- This consists of all objects available to the player for predicting object placement, which can be arranged based on ID.
 
 local maximumValueDimensionIndexArray = TensorL2D:findMaximumValueDimensionIndexArray(probabilityVector)
 
 local topID = maximumValueDimensionIndexArray[1] -- We only need the maximum from rows since there is only one column generated here.
 
 displayObject(Player, topID)

end

```

## Conclusion

This tutorial showed you on how to create object placement prediction model that allows you to aid your players' construction process. All you need is some data, some models and a bit of practice to get this right!

That's all for today and see you later!
