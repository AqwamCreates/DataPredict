local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local ReinforcementLearningBaseModel = require(script.Parent.ReinforcementLearningBaseModel)

DeepDoubleQLearningModel = {}

DeepDoubleQLearningModel.__index = DeepDoubleQLearningModel

setmetatable(DeepDoubleQLearningModel, ReinforcementLearningBaseModel)

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

function DeepDoubleQLearningModel.new(discountFactor)

	local NewDeepDoubleQLearningModel = ReinforcementLearningBaseModel.new(discountFactor)

	setmetatable(NewDeepDoubleQLearningModel, DeepDoubleQLearningModel)
	
	NewDeepDoubleQLearningModel.ModelParametersArray = {}
	
	NewDeepDoubleQLearningModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local Model = NewDeepDoubleQLearningModel.Model
		
		local randomProbability = Random.new():NextNumber()

		local updateSecondModel = (randomProbability >= 0.5)

		local selectedModelNumberForTargetVector = (updateSecondModel and 1) or 2

		local selectedModelNumberForUpdate = (updateSecondModel and 2) or 1

		NewDeepDoubleQLearningModel:loadModelParametersFromModelParametersArray(selectedModelNumberForTargetVector)

		local lossVector, temporalDifferenceError = NewDeepDoubleQLearningModel:generateLossVector(previousFeatureVector, action, rewardValue, currentFeatureVector)

		NewDeepDoubleQLearningModel:saveModelParametersFromModelParametersArray(selectedModelNumberForTargetVector)

		NewDeepDoubleQLearningModel:loadModelParametersFromModelParametersArray(selectedModelNumberForUpdate)

		Model:forwardPropagate(previousFeatureVector, true)
		
		Model:backPropagate(lossVector, true)

		NewDeepDoubleQLearningModel:saveModelParametersFromModelParametersArray(selectedModelNumberForUpdate)
		
		return temporalDifferenceError
		
	end)

	return NewDeepDoubleQLearningModel

end

function DeepDoubleQLearningModel:setParameters(discountFactor)

	self.discountFactor = discountFactor or self.discountFactor

end

function DeepDoubleQLearningModel:saveModelParametersFromModelParametersArray(index)

	self.ModelParametersArray[index] = self.Model:getModelParameters()

end

function DeepDoubleQLearningModel:loadModelParametersFromModelParametersArray(index)
	
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

function DeepDoubleQLearningModel:generateLossVector(previousFeatureVector, action, rewardValue, currentFeatureVector)
	
	local Model = self.Model

	local predictedValue, maxQValue = Model:predict(currentFeatureVector)

	local targetValue = rewardValue + (self.discountFactor * maxQValue[1][1])
	
	local ClassesList = Model:getClassesList()
	
	local numberOfClasses = #ClassesList

	local previousVector = Model:predict(previousFeatureVector, true)

	local actionIndex = table.find(ClassesList, action)
	
	local lastValue = previousVector[1][actionIndex]
	
	local temporalDifferenceError = targetValue - lastValue
		
	local lossVector = AqwamMatrixLibrary:createMatrix(1, numberOfClasses, 0)

	lossVector[1][actionIndex] = temporalDifferenceError
	
	return lossVector, temporalDifferenceError
	
end

function DeepDoubleQLearningModel:setModelParameters1(ModelParameters1, doNotDeepCopy)
	
	if (doNotDeepCopy) then
		
		self.ModelParametersArray[1] = ModelParameters1
		
	else
		
		self.ModelParametersArray[1] = deepCopyTable(ModelParameters1)
		
	end

end

function DeepDoubleQLearningModel:setModelParameters2(ModelParameters2, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.ModelParametersArray[2] = ModelParameters2

	else

		self.ModelParametersArray[2] = deepCopyTable(ModelParameters2)

	end

end

function DeepDoubleQLearningModel:getModelParameters1(doNotDeepCopy)
	
	if (doNotDeepCopy) then
		
		return self.ModelParametersArray[1]
		
	else
		
		return deepCopyTable(self.ModelParametersArray[1])
		
	end

end

function DeepDoubleQLearningModel:getModelParameters2(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.ModelParametersArray[2]

	else

		return deepCopyTable(self.ModelParametersArray[2])

	end

end

return DeepDoubleQLearningModel
