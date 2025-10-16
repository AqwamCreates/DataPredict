# Safe Practices For Live Projects

Under this section, the code shown below demonstrates on how you detect defective model parameters before you can produce prediction and perform a rollback. We will show you two methods of doing this:

* Method 1: Using ModelSafeguardWrapper

* Method 2: Using Manual

### Method 1: Using ModelSafeguardWrapper

ModelSafeguardWrapper contains useful modifications that automatically restores old model parameters upon detecting problematic cost value and removal of defective data. Below, we will show you how to create this object.

```lua

local SafeguardedModel = DataPredict.Others.ModelSafeguardWrapper.new({Model = Model})

local costArray = SafeguardedModel:train(featureMatrix, labelVector) -- You can now call train() with the added advantage of model parameters rollback.

local predictedLabelVector = SafeguardedModel:predict(featureMatrix) -- You can also use predict() as usual without grabbing the original model.

```

### Method 2: Using Manual

```lua

-- Before you train or update anything, ensure that you keep the original model parameters.

-- Ensure that we want the model to do "deep copy" of the model parameters.

local ModelParameters = Model:getModelParameters(false)

local canUseModel = false -- This flag is used to ensure that the model is not performing prediction elsewhere.

local function checkIfIsAcceptableValue(value)

    return (value == value) and (value ~= math.huge) and (value ~= -math.huge) and (type(value) == "number")

end

local function filterOutDefectiveData(dataMatrix)

    local rowToDeleteArray = {}

    for i, dataVector in ipairs(dataMatrix) do

      for j, value in ipairs(dataVector) do

          if (checkIfIsAcceptableValue(value)) then continue end

          table.insert(rowToDeleteArray, i)

          break

      end

    end

  local filteredDataMatrix = {}

  for i = #dataMatrix, 1, -1 do

    if (table.find(rowToDeleteArray, i)) then continue end

    table.insert(filteredDataMatrix, dataMatrix[i])
  
  end

  return filteredDataMatrix

end

while true do

   -- Notice that we have cost array being assigned to it. This will be one of the way on detecting defective model parameters.

  local costArray = Model:train(featureDataMatrix, labelVector)

  -- Check if the final element is a "nan", "inf" or "nil" value.

  local finalCostValue = costArray[#costArray]

  -- If the final cost value is not any of them, we can safely break out from this loop

    if (checkIfIsAcceptableValue(finalCostValue)) then

        canUseModel = true
    
        break

    end

  -- Otherwise, restore immediately.

 -- Ensure that we want the model to do "deep copy" of the model parameters.

  Model:setModelParameters(false)

  -- You may need to scan the data for issues.

  featureDataMatrix = filterOutDefectiveData(labelVector)

  labelVector = filterOutDefectiveData(labelVector)

end

```
