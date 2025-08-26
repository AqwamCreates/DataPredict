--[[

	--------------------------------------------------------------------

	Aqwam's Machine, Deep And Reinforcement Learning Library (DataPredict)

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

local DeepReinforcementLearningBaseModel = require(script.Parent.DeepReinforcementLearningBaseModel)

DeepClippedDoubleQLearningModel = {}

DeepClippedDoubleQLearningModel.__index = DeepClippedDoubleQLearningModel

setmetatable(DeepClippedDoubleQLearningModel, DeepReinforcementLearningBaseModel)

local defaultLambda = 0

function DeepClippedDoubleQLearningModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewDeepClippedDoubleQLearningModel = DeepReinforcementLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewDeepClippedDoubleQLearningModel, DeepClippedDoubleQLearningModel)
	
	NewDeepClippedDoubleQLearningModel:setName("DeepClippedDoubleQLearning")

	NewDeepClippedDoubleQLearningModel.ModelParametersArray = {}

	NewDeepClippedDoubleQLearningModel.EligibilityTrace = parameterDictionary.EligibilityTrace
	
	NewDeepClippedDoubleQLearningModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)
		
		local Model = NewDeepClippedDoubleQLearningModel.Model
		
		local discountFactor = NewDeepClippedDoubleQLearningModel.discountFactor
		
		local ModelParametersArray = NewDeepClippedDoubleQLearningModel.ModelParametersArray
		
		local EligibilityTrace = NewDeepClippedDoubleQLearningModel.EligibilityTrace

		local maxQValueArray = {}

		for i = 1, 2, 1 do

			Model:setModelParameters(ModelParametersArray[i], true)

			local _, maxQValue = Model:predict(currentFeatureVector)

			table.insert(maxQValueArray, maxQValue[1][1])
			
			ModelParametersArray[i] = Model:getModelParameters(true)

		end

		local maxQValue = math.min(table.unpack(maxQValueArray))

		local targetValue = rewardValue + (discountFactor * (1 - terminalStateValue) * maxQValue)
		
		local ClassesList = Model:getClassesList()

		local actionIndex = table.find(ClassesList, action)
		
		local numberOfClasses = #ClassesList
		
		local outputDimensionSizeArray = {1, numberOfClasses}
		
		local temporalDifferenceErrorVector = AqwamTensorLibrary:createTensor({1, 2})
		
		if (EligibilityTrace) then
			
			EligibilityTrace:increment(actionIndex, discountFactor, outputDimensionSizeArray)

		end

		for i = 1, 2, 1 do

			Model:setModelParameters(ModelParametersArray[i], true)

			local previousVector = Model:forwardPropagate(previousFeatureVector, true)

			local lastValue = previousVector[1][actionIndex]
			
			local temporalDifferenceError = targetValue - lastValue
			
			local lossVector = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)

			lossVector[1][actionIndex] = temporalDifferenceError
			
			temporalDifferenceErrorVector[1][i] = temporalDifferenceError
			
			if (EligibilityTrace) then lossVector = EligibilityTrace:increment(lossVector) end
			
			local negatedLossVector = AqwamTensorLibrary:unaryMinus(lossVector) -- The original non-deep Q-Learning version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error vector to make the neural network to perform gradient ascent.
			
			Model:update(negatedLossVector, true)
			
			ModelParametersArray[i] = Model:getModelParameters(true)

		end
		
		return temporalDifferenceErrorVector

	end)
	
	NewDeepClippedDoubleQLearningModel:setEpisodeUpdateFunction(function(terminalStateValue) 
		
		NewDeepClippedDoubleQLearningModel.EligibilityTrace:reset()
		
	end)

	NewDeepClippedDoubleQLearningModel:setResetFunction(function() 
		
		NewDeepClippedDoubleQLearningModel.EligibilityTrace:reset()
		
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
