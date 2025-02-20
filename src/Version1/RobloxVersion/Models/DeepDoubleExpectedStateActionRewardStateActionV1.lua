--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	Email: aqwam.harish.aiman@gmail.com
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------
	
	DO NOT REMOVE THIS TEXT!
	
	--------------------------------------------------------------------

--]]

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local ReinforcementLearningBaseModel = require(script.Parent.ReinforcementLearningBaseModel)

DeepDoubleExpectedStateActionRewardStateActionModel = {}

DeepDoubleExpectedStateActionRewardStateActionModel.__index = DeepDoubleExpectedStateActionRewardStateActionModel

setmetatable(DeepDoubleExpectedStateActionRewardStateActionModel, ReinforcementLearningBaseModel)

local defaultEpsilon = 0.5

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

function DeepDoubleExpectedStateActionRewardStateActionModel.new(epsilon, discountFactor)

	local NewDeepDoubleExpectedStateActionRewardStateActionModel = ReinforcementLearningBaseModel.new(discountFactor)
	
	NewDeepDoubleExpectedStateActionRewardStateActionModel.epsilon = epsilon or defaultEpsilon
	
	setmetatable(NewDeepDoubleExpectedStateActionRewardStateActionModel, DeepDoubleExpectedStateActionRewardStateActionModel)

	NewDeepDoubleExpectedStateActionRewardStateActionModel.ModelParametersArray = {}

	NewDeepDoubleExpectedStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local Model = NewDeepDoubleExpectedStateActionRewardStateActionModel.Model

		local randomProbability = math.random()

		local updateSecondModel = (randomProbability >= 0.5)

		local selectedModelNumberForTargetVector = (updateSecondModel and 1) or 2

		local selectedModelNumberForUpdate = (updateSecondModel and 2) or 1

		local lossVector, temporalDifferenceError = NewDeepDoubleExpectedStateActionRewardStateActionModel:generateLossVector(previousFeatureVector, action, rewardValue, currentFeatureVector, selectedModelNumberForTargetVector, selectedModelNumberForUpdate)

		Model:forwardPropagate(previousFeatureVector, true, true)

		Model:backwardPropagate(lossVector, true)

		NewDeepDoubleExpectedStateActionRewardStateActionModel:saveModelParametersFromModelParametersArray(selectedModelNumberForUpdate)
		
		return temporalDifferenceError

	end)
	
	NewDeepDoubleExpectedStateActionRewardStateActionModel:setEpisodeUpdateFunction(function() end)

	NewDeepDoubleExpectedStateActionRewardStateActionModel:setResetFunction(function() end)

	return NewDeepDoubleExpectedStateActionRewardStateActionModel

end

function DeepDoubleExpectedStateActionRewardStateActionModel:setParameters(epsilon, discountFactor)

	self.epsilon = epsilon or self.epsilon

	self.discountFactor = discountFactor or self.discountFactor

end

function DeepDoubleExpectedStateActionRewardStateActionModel:saveModelParametersFromModelParametersArray(index)

	self.ModelParametersArray[index] = self.Model:getModelParameters()

end

function DeepDoubleExpectedStateActionRewardStateActionModel:loadModelParametersFromModelParametersArray(index)
	
	local Model = self.Model

	if (not self.ModelParametersArray[1]) and (not self.ModelParametersArray[2]) then

		Model:generateLayers()

		self:saveModelParametersFromModelParametersArray(1)

		self:saveModelParametersFromModelParametersArray(2)

	end

	local CurrentModelParameters = self.ModelParametersArray[index]

	Model:setModelParameters(CurrentModelParameters, true)

end

function DeepDoubleExpectedStateActionRewardStateActionModel:generateLossVector(previousFeatureVector, action, rewardValue, currentFeatureVector, selectedModelNumberForTargetVector, selectedModelNumberForUpdate)
	
	local Model = self.Model

	local expectedQValue = 0

	local numberOfGreedyActions = 0
	
	local ClassesList = Model:getClassesList()

	local numberOfActions = #ClassesList

	local actionIndex = table.find(ClassesList, action)
	
	self:loadModelParametersFromModelParametersArray(selectedModelNumberForUpdate)

	local previousVector = Model:forwardPropagate(previousFeatureVector)
	
	self:loadModelParametersFromModelParametersArray(selectedModelNumberForTargetVector)

	local targetVector = Model:forwardPropagate(currentFeatureVector)

	local maxQValue = targetVector[1][actionIndex]

	for i = 1, numberOfActions, 1 do

		if (targetVector[1][i] ~= maxQValue) then continue end

		numberOfGreedyActions = numberOfGreedyActions + 1

	end

	local nonGreedyActionProbability = self.epsilon / numberOfActions

	local greedyActionProbability = ((1 - self.epsilon) / numberOfGreedyActions) + nonGreedyActionProbability

	for _, qValue in ipairs(targetVector[1]) do

		if (qValue == maxQValue) then

			expectedQValue = expectedQValue + (qValue * greedyActionProbability)

		else

			expectedQValue = expectedQValue + (qValue * nonGreedyActionProbability)

		end

	end

	local targetValue = rewardValue + (self.discountFactor * expectedQValue)
	
	local lastValue = previousVector[1][actionIndex]

	local temporalDifferenceError = targetValue - lastValue

	local lossVector = AqwamMatrixLibrary:createMatrix(1, numberOfActions, 0)

	lossVector[1][actionIndex] = temporalDifferenceError

	return lossVector, temporalDifferenceError

end

function DeepDoubleExpectedStateActionRewardStateActionModel:setModelParameters1(ModelParameters1, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.ModelParametersArray[1] = ModelParameters1

	else

		self.ModelParametersArray[1] = deepCopyTable(ModelParameters1)

	end

end

function DeepDoubleExpectedStateActionRewardStateActionModel:setModelParameters2(ModelParameters2, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.ModelParametersArray[2] = ModelParameters2

	else

		self.ModelParametersArray[2] = deepCopyTable(ModelParameters2)

	end

end

function DeepDoubleExpectedStateActionRewardStateActionModel:getModelParameters1(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.ModelParametersArray[1]

	else

		return deepCopyTable(self.ModelParametersArray[1])

	end

end

function DeepDoubleExpectedStateActionRewardStateActionModel:getModelParameters2(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.ModelParametersArray[2]

	else

		return deepCopyTable(self.ModelParametersArray[2])

	end

end

return DeepDoubleExpectedStateActionRewardStateActionModel