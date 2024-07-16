local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local ReinforcementLearningBaseModel = require(script.Parent.ReinforcementLearningBaseModel)

DeepDoubleStateActionRewardStateActionModel = {}

DeepDoubleStateActionRewardStateActionModel.__index = DeepDoubleStateActionRewardStateActionModel

setmetatable(DeepDoubleStateActionRewardStateActionModel, ReinforcementLearningBaseModel)

local function deepCopyTable(original, copies)

	copies = copies or {}

	local originalType = type(original)

	local copy

	if (originalType == 'table') then

		if copies[original] then

			copy = copies[original]

		else

			copy = {}

			copies[original] = copy

			for originalKey, originalValue in next, original, nil do

				copy[deepCopyTable(originalKey, copies)] = deepCopyTable(originalValue, copies)

			end

			setmetatable(copy, deepCopyTable(getmetatable(original), copies))

		end

	else

		copy = original

	end

	return copy

end

function DeepDoubleStateActionRewardStateActionModel.new(discountFactor)

	local NewDeepDoubleStateActionRewardStateActionModel = ReinforcementLearningBaseModel.new(discountFactor)

	setmetatable(NewDeepDoubleStateActionRewardStateActionModel, DeepDoubleStateActionRewardStateActionModel)

	NewDeepDoubleStateActionRewardStateActionModel.ModelParametersArray = {}

	NewDeepDoubleStateActionRewardStateActionModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local Model = NewDeepDoubleStateActionRewardStateActionModel.Model

		local randomProbability = math.random()

		local updateSecondModel = (randomProbability >= 0.5)

		local selectedModelNumberForTargetVector = (updateSecondModel and 1) or 2

		local selectedModelNumberForUpdate = (updateSecondModel and 2) or 1

		NewDeepDoubleStateActionRewardStateActionModel:loadModelParametersFromModelParametersArray(selectedModelNumberForTargetVector)

		local lossVector = NewDeepDoubleStateActionRewardStateActionModel:generateLossVector(previousFeatureVector, action, rewardValue, currentFeatureVector)

		NewDeepDoubleStateActionRewardStateActionModel:saveModelParametersFromModelParametersArray(selectedModelNumberForTargetVector)

		NewDeepDoubleStateActionRewardStateActionModel:loadModelParametersFromModelParametersArray(selectedModelNumberForUpdate)

		Model:forwardPropagate(previousFeatureVector, true)
		
		Model:backPropagate(lossVector, true)

		NewDeepDoubleStateActionRewardStateActionModel:saveModelParametersFromModelParametersArray(selectedModelNumberForUpdate)
		
		return lossVector

	end)

	return NewDeepDoubleStateActionRewardStateActionModel

end

function DeepDoubleStateActionRewardStateActionModel:setParameters(discountFactor)

	self.discountFactor =  discountFactor or self.discountFactor

end

function DeepDoubleStateActionRewardStateActionModel:saveModelParametersFromModelParametersArray(index)

	self.ModelParametersArray[index] = self.Model:getModelParameters()

end

function DeepDoubleStateActionRewardStateActionModel:loadModelParametersFromModelParametersArray(index)
	
	local Model = self.Model

	local FirstModelParameters = self.ModelParametersArray[1]

	local SecondModelParameters = self.ModelParametersArray[2]

	if (FirstModelParameters == nil) and (SecondModelParameters == nil) then

		Model:generateLayers()

		self:saveModelParametersFromModelParametersArray(1)

		self:saveModelParametersFromModelParametersArray(2)

	end

	local CurrentModelParameters = self.ModelParametersArray[index]

	Model:setModelParameters(CurrentModelParameters, true)

end

function DeepDoubleStateActionRewardStateActionModel:generateLossVector(previousFeatureVector, action, rewardValue, currentFeatureVector)
	
	local Model = self.Model

	local targetVector = Model:predict(currentFeatureVector, true)

	local dicountedTargetVector = AqwamMatrixLibrary:multiply(self.discountFactor, targetVector)

	local newTargetVector = AqwamMatrixLibrary:add(rewardValue, dicountedTargetVector)
	
	local previousVector = Model:predict(previousFeatureVector, true)
	
	local lossVector = AqwamMatrixLibrary:subtract(newTargetVector, previousVector)

	return lossVector

end

function DeepDoubleStateActionRewardStateActionModel:setModelParameters1(ModelParameters1, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.ModelParametersArray[1] = ModelParameters1

	else

		self.ModelParametersArray[1] = deepCopyTable(ModelParameters1)

	end

end

function DeepDoubleStateActionRewardStateActionModel:setModelParameters2(ModelParameters2, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.ModelParametersArray[2] = ModelParameters2

	else

		self.ModelParametersArray[2] = deepCopyTable(ModelParameters2)

	end

end

function DeepDoubleStateActionRewardStateActionModel:getModelParameters1(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.ModelParametersArray[1]

	else

		return deepCopyTable(self.ModelParametersArray[1])

	end

end

function DeepDoubleStateActionRewardStateActionModel:getModelParameters2(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.ModelParametersArray[2]

	else

		return deepCopyTable(self.ModelParametersArray[2])

	end

end

return DeepDoubleStateActionRewardStateActionModel
