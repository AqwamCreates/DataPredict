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

DeepQLearningModel = {}

DeepQLearningModel.__index = DeepQLearningModel

setmetatable(DeepQLearningModel, DeepReinforcementLearningBaseModel)

function DeepQLearningModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewDeepQLearningModel = DeepReinforcementLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewDeepQLearningModel, DeepQLearningModel)
	
	NewDeepQLearningModel:setName("DeepQLearning")
	
	NewDeepQLearningModel.EligibilityTrace = parameterDictionary.EligibilityTrace
	
	NewDeepQLearningModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)
		
		local Model = NewDeepQLearningModel.Model
		
		local discountFactor = NewDeepQLearningModel.discountFactor
		
		local EligibilityTrace = NewDeepQLearningModel.EligibilityTrace

		local _, maxQValue = Model:predict(currentFeatureVector)

		local targetValue = rewardValue + (discountFactor * (1 - terminalStateValue) * maxQValue[1][1])
		
		local ClassesList = Model:getClassesList()

		local numberOfClasses = #ClassesList

		local previousVector = Model:forwardPropagate(previousFeatureVector)

		local actionIndex = table.find(ClassesList, action)

		local lastValue = previousVector[1][actionIndex]

		local temporalDifferenceError = targetValue - lastValue
		
		local outputDimensionSizeArray = {1, numberOfClasses}

		local temporalDifferenceErrorVector = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)

		temporalDifferenceErrorVector[1][actionIndex] = temporalDifferenceError
		
		if (EligibilityTrace) then

			EligibilityTrace:increment(actionIndex, discountFactor, outputDimensionSizeArray)

			temporalDifferenceErrorVector = EligibilityTrace:calculate(temporalDifferenceErrorVector)

		end
		
		local negatedTemporalDifferenceErrorVector = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorVector) -- The original non-deep Q-Learning version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error vector to make the neural network to perform gradient ascent.
		
		Model:forwardPropagate(previousFeatureVector, true)

		Model:update(negatedTemporalDifferenceErrorVector, true)
		
		return temporalDifferenceErrorVector

	end)
	
	NewDeepQLearningModel:setEpisodeUpdateFunction(function(terminalStateValue)
		
		NewDeepQLearningModel.EligibilityTrace:reset()
		
	end)

	NewDeepQLearningModel:setResetFunction(function()
		
		NewDeepQLearningModel.EligibilityTrace:reset()
		
	end)

	return NewDeepQLearningModel

end

return DeepQLearningModel
