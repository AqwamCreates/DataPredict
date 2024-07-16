--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

DeepDoubleDuelingQLearning = {}

DeepDoubleDuelingQLearning.__index = DeepDoubleDuelingQLearning

local defaultDiscountFactor = 0.95

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

function DeepDoubleDuelingQLearning.new(discountFactor)

	local NewDeepDuelingQLearning = {}

	setmetatable(NewDeepDuelingQLearning, DeepDoubleDuelingQLearning)

	NewDeepDuelingQLearning.discountFactor = discountFactor or defaultDiscountFactor
	
	NewDeepDuelingQLearning.AdvantageModelParametersArray = {}
	
	NewDeepDuelingQLearning.ValueModelParametersArray = {}

	return NewDeepDuelingQLearning

end

function DeepDoubleDuelingQLearning:setParameters(discountFactor)

	self.discountFactor = discountFactor or self.discountFactor

end

function DeepDoubleDuelingQLearning:setAdvantageModel(Model)

	self.AdvantageModel = Model

end

function DeepDoubleDuelingQLearning:setValueModel(Model)

	self.ValueModel = Model

end

function DeepDoubleDuelingQLearning:forwardPropagate(featureVector)

	local vValue = self.ValueModel:predict(featureVector, true)[1][1]

	local advantageMatrix = self.AdvantageModel:predict(featureVector, true)

	local meanAdvantageVector = AqwamMatrixLibrary:horizontalMean(advantageMatrix)

	local qValueVectorPart1 = AqwamMatrixLibrary:subtract(advantageMatrix, meanAdvantageVector)

	local qValueVector = AqwamMatrixLibrary:add(vValue, qValueVectorPart1)

	return qValueVector, vValue

end

function DeepDoubleDuelingQLearning:generateLoss(previousFeatureVector, action, rewardValue, currentFeatureVector)

	local previousQValue, previousVValue = self:forwardPropagate(previousFeatureVector)

	local currentQValueVector, currentVValue = self:forwardPropagate(currentFeatureVector)

	local ClassesList = self.AdvantageModel:getClassesList()

	local actionIndex = table.find(ClassesList, action)

	local maxCurrentQValue = currentQValueVector[1][actionIndex]

	local expectedQValue = rewardValue + (self.discountFactor * maxCurrentQValue)

	local qLossVector = AqwamMatrixLibrary:subtract(expectedQValue, previousQValue)

	local vLoss = currentVValue - previousVValue

	return qLossVector, vLoss

end

function DeepDoubleDuelingQLearning:update(previousFeatureVector, action, rewardValue, currentFeatureVector)
	
	local AdvantageModel = self.AdvantageModel
	
	local ValueModel = self.ValueModel
	
	local randomProbability = Random.new():NextNumber()

	local updateSecondModel = (randomProbability >= 0.5)

	local selectedModelNumberForTargetVector = (updateSecondModel and 1) or 2

	local selectedModelNumberForUpdate = (updateSecondModel and 2) or 1
	
	self:loadAdvantageModelParametersFromAdvantageModelParametersArray(selectedModelNumberForTargetVector)
	
	self:loadValueModelParametersFromValueModelParametersArray(selectedModelNumberForTargetVector)

	local qLossVector, vLoss = self:generateLoss(previousFeatureVector, action, rewardValue, currentFeatureVector)
	
	self:saveAdvantageModelParametersFromAdvantageModelParametersArray(selectedModelNumberForTargetVector)
	
	self:saveValueModelParametersFromValueModelParametersArray(selectedModelNumberForTargetVector)

	self:loadAdvantageModelParametersFromAdvantageModelParametersArray(selectedModelNumberForUpdate)
	
	self:loadValueModelParametersFromValueModelParametersArray(selectedModelNumberForUpdate)
	
	AdvantageModel:forwardPropagate(previousFeatureVector, true)

	AdvantageModel:backPropagate(qLossVector, true)

	ValueModel:forwardPropagate(previousFeatureVector, true)

	ValueModel:backPropagate(vLoss, true)

	return vLoss

end

function DeepDoubleDuelingQLearning:saveAdvantageModelParametersFromAdvantageModelParametersArray(index)

	self.AdvantageModelParametersArray[index] = self.AdvantageModel:getModelParameters()

end

function DeepDoubleDuelingQLearning:saveValueModelParametersFromValueModelParametersArray(index)

	self.ValueModelParametersArray[index] = self.ValueModel:getModelParameters()

end

function DeepDoubleDuelingQLearning:loadAdvantageModelParametersFromAdvantageModelParametersArray(index)

	local AdvantageModel = self.AdvantageModel

	local FirstModelParameters = self.AdvantageModelParametersArray[1]

	local SecondModelParameters = self.AdvantageModelParametersArray[2]

	if (FirstModelParameters == nil) and (SecondModelParameters == nil) then

		AdvantageModel:generateLayers()

		self:saveAdvantageModelParametersFromAdvantageModelParametersArray(1)

		self:saveAdvantageModelParametersFromAdvantageModelParametersArray(2)

	end

	local CurrentModelParameters = self.AdvantageModelParametersArray[index]

	AdvantageModel:setModelParameters(CurrentModelParameters, true)

end

function DeepDoubleDuelingQLearning:loadValueModelParametersFromValueModelParametersArray(index)

	local ValueModel = self.ValueModel

	local FirstModelParameters = self.ValueModelParametersArray[1]

	local SecondModelParameters = self.ValueModelParametersArray[2]

	if (FirstModelParameters == nil) and (SecondModelParameters == nil) then

		ValueModel:generateLayers()

		self:saveValueModelParametersFromValueModelParametersArray(1)

		self:saveValueModelParametersFromValueModelParametersArray(2)

	end

	local CurrentModelParameters = self.ValueModelParametersArray[index]

	ValueModel:setModelParameters(CurrentModelParameters, true)

end

function DeepDoubleDuelingQLearning:predict(featureVector, returnOriginalOutput)

	return self.AdvantageModel:predict(featureVector, returnOriginalOutput)

end

function DeepDoubleDuelingQLearning:episodeUpdate()
	
	
end

function DeepDoubleDuelingQLearning:reset()
	
	
end

function DeepDoubleDuelingQLearning:setAdvantageModelParameters1(AdvantageModelParameters1, doNotDeepCopy)
	
	if (doNotDeepCopy) then
		
		self.AdvantageModelParametersArray[1] = deepCopyTable(AdvantageModelParameters1)
		
	else
		
		self.AdvantageModelParametersArray[1] = AdvantageModelParameters1
		
	end

end

function DeepDoubleDuelingQLearning:setAdvantageModelParameters2(AdvantageModelParameters2, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.AdvantageModelParametersArray[2] = deepCopyTable(AdvantageModelParameters2)

	else

		self.AdvantageModelParametersArray[2] = AdvantageModelParameters2

	end

end

function DeepDoubleDuelingQLearning:setValueModelParameters1(ValueModelParameters1, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.ValueModelParametersArray[1] = deepCopyTable(ValueModelParameters1)

	else

		self.ValueModelParametersArray[1] = ValueModelParameters1

	end

end

function DeepDoubleDuelingQLearning:setValueModelParameters2(ValueModelParameters2, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.ValueModelParametersArray[2] = deepCopyTable(ValueModelParameters2)

	else

		self.ValueModelParametersArray[2] = ValueModelParameters2

	end

end

function DeepDoubleDuelingQLearning:getAdvantageModelParameters1(doNotDeepCopy)
	
	if (doNotDeepCopy) then
		
		return self.AdvantageModelParametersArray[1]
		
	else
		
		return deepCopyTable(self.AdvantageModelParametersArray[1])
		
	end

end

function DeepDoubleDuelingQLearning:getAdvantageModelParameters2(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.AdvantageModelParametersArray[2]

	else

		return deepCopyTable(self.AdvantageModelParametersArray[2])

	end

end

function DeepDoubleDuelingQLearning:getValueModelParameters1(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.ValueModelParametersArray[1]

	else

		return deepCopyTable(self.ValueModelParametersArray[1])

	end

end

function DeepDoubleDuelingQLearning:getValueModelParameters2(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.ValueModelParametersArray[2]

	else

		return deepCopyTable(self.ValueModelParametersArray[2])

	end

end

function DeepDoubleDuelingQLearning:destroy()

	setmetatable(self, nil)

	table.clear(self)

	self = nil

end

return DeepDoubleDuelingQLearning
