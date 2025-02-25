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

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local ReinforcementLearningBaseModel = require(script.Parent.ReinforcementLearningBaseModel)

DeepClippedDoubleQLearningModel = {}

DeepClippedDoubleQLearningModel.__index = DeepClippedDoubleQLearningModel

setmetatable(DeepClippedDoubleQLearningModel, ReinforcementLearningBaseModel)

local defaultLambda = 0

function DeepClippedDoubleQLearningModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewDeepClippedDoubleQLearningModel = ReinforcementLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewDeepClippedDoubleQLearningModel, DeepClippedDoubleQLearningModel)
	
	NewDeepClippedDoubleQLearningModel:setName("DeepClippedDoubleQLearning")

	NewDeepClippedDoubleQLearningModel.ModelParametersArray = {}
	
	NewDeepClippedDoubleQLearningModel.lambda = parameterDictionary.lambda or defaultLambda

	NewDeepClippedDoubleQLearningModel.eligibilityTrace = parameterDictionary.eligibilityTrace 
	
	NewDeepClippedDoubleQLearningModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)
		
		local Model = NewDeepClippedDoubleQLearningModel.Model

		local maxQValueArray = {}

		for i = 1, 2, 1 do

			Model:setModelParameters(NewDeepClippedDoubleQLearningModel.ModelParametersArray[i], true)

			local _, maxQValue = Model:predict(currentFeatureVector)

			table.insert(maxQValueArray, maxQValue[1][1])

		end

		local maxQValue = math.min(table.unpack(maxQValueArray))

		local targetValue = rewardValue + (NewDeepClippedDoubleQLearningModel.discountFactor * (1 - terminalStateValue) * maxQValue)
		
		local ClassesList = Model:getClassesList()

		local actionIndex = table.find(ClassesList, action)
		
		local numberOfClasses = #ClassesList
		
		local outputDimensionSizeArray = {1, numberOfClasses}
		
		local eligibilityTrace = NewDeepClippedDoubleQLearningModel.eligibilityTrace
		
		local temporalDifferenceErrorVector = AqwamTensorLibrary:createTensor(outputDimensionSizeArray)
		
		temporalDifferenceErrorVector[1][actionIndex] = ((targetValue - maxQValueArray[1]) + (targetValue - maxQValueArray[2])) / 2
		
		if (NewDeepClippedDoubleQLearningModel.lambda ~= 0) then

			if (not eligibilityTrace) then eligibilityTrace = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0) end

			eligibilityTrace = AqwamTensorLibrary:multiply(eligibilityTrace, NewDeepClippedDoubleQLearningModel.discountFactor * NewDeepClippedDoubleQLearningModel.lambda)

			eligibilityTrace[1][actionIndex] = eligibilityTrace[1][actionIndex] + 1
			
			NewDeepClippedDoubleQLearningModel.eligibilityTrace = eligibilityTrace

		end

		for i = 1, 2, 1 do

			Model:setModelParameters(NewDeepClippedDoubleQLearningModel.ModelParametersArray[i], true)

			local previousVector = Model:forwardPropagate(previousFeatureVector, true)

			local lastValue = previousVector[1][actionIndex]
			
			local temporalDifferenceError = targetValue - lastValue
			
			local lossVector = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)

			lossVector[1][actionIndex] = temporalDifferenceError
			
			if (NewDeepClippedDoubleQLearningModel.lambda ~= 0) then lossVector = AqwamTensorLibrary:multiply(lossVector, eligibilityTrace) end
			
			Model:backwardPropagate(temporalDifferenceErrorVector, true)

		end
		
		return temporalDifferenceErrorVector

	end)
	
	NewDeepClippedDoubleQLearningModel:setEpisodeUpdateFunction(function(terminalStateValue) 
		
		NewDeepClippedDoubleQLearningModel.eligibilityTrace = nil
		
	end)

	NewDeepClippedDoubleQLearningModel:setResetFunction(function() 
		
		NewDeepClippedDoubleQLearningModel.eligibilityTrace = nil
		
	end)

	return NewDeepClippedDoubleQLearningModel

end

function DeepClippedDoubleQLearningModel:setModelParameters1(ModelParameters1, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.ModelParametersArray[1] = ModelParameters1

	else

		self.ModelParametersArray[1] = self:deepCopyTable(ModelParameters1)

	end

end

function DeepClippedDoubleQLearningModel:setModelParameters2(ModelParameters2, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.ModelParametersArray[2] = ModelParameters2

	else

		self.ModelParametersArray[2] = self:deepCopyTable(ModelParameters2)

	end

end

function DeepClippedDoubleQLearningModel:getModelParameters1(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.ModelParametersArray[1]

	else

		return self:deepCopyTable(self.ModelParametersArray[1])

	end

end

function DeepClippedDoubleQLearningModel:getModelParameters2(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.ModelParametersArray[2]

	else

		return self:deepCopyTable(self.ModelParametersArray[2])

	end

end

return DeepClippedDoubleQLearningModel