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

local DeepDoubleStateActionRewardStateActionModel = {}

DeepDoubleStateActionRewardStateActionModel.__index = DeepDoubleStateActionRewardStateActionModel

setmetatable(DeepDoubleStateActionRewardStateActionModel, DeepReinforcementLearningBaseModel)

local defaultAveragingRate = 0.01

local function rateAverageModelParameters(averagingRate, TargetModelParameters, PrimaryModelParameters)

	local averagingRateComplement = 1 - averagingRate

	for layer = 1, #TargetModelParameters, 1 do

		local PrimaryModelParametersPart = AqwamTensorLibrary:multiply(averagingRate, PrimaryModelParameters[layer])

		local TargetModelParametersPart = AqwamTensorLibrary:multiply(averagingRateComplement, TargetModelParameters[layer])

		TargetModelParameters[layer] = AqwamTensorLibrary:add(PrimaryModelParametersPart, TargetModelParametersPart)

	end

	return TargetModelParameters

end

function DeepDoubleStateActionRewardStateActionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewDeepDoubleStateActionRewardStateActionModel = DeepReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewDeepDoubleStateActionRewardStateActionModel, DeepDoubleStateActionRewardStateActionModel)
	
	NewDeepDoubleStateActionRewardStateActionModel:setName("DeepDoubleStateActionRewardStateActionV2")

	NewDeepDoubleStateActionRewardStateActionModel.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate

	NewDeepDoubleStateActionRewardStateActionModel.EligibilityTrace = parameterDictionary.EligibilityTrace

	NewDeepDoubleStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousFeatureVector, previousAction, rewardValue, currentFeatureVector, currentAction, terminalStateValue)
		
		local Model = NewDeepDoubleStateActionRewardStateActionModel.Model
		
		local discountFactor = NewDeepDoubleStateActionRewardStateActionModel.discountFactor

		local EligibilityTrace = NewDeepDoubleStateActionRewardStateActionModel.EligibilityTrace
		
		local PrimaryModelParameters = Model:getModelParameters(true)

		if (PrimaryModelParameters) then 
			
			Model:generateLayers()
			
			PrimaryModelParameters = Model:getModelParameters(true)
			
		end
		
		local currentQVector = Model:forwardPropagate(currentFeatureVector)

		local previousQVector = Model:forwardPropagate(previousFeatureVector)

		local ClassesList = Model:getClassesList()

		local numberOfClasses = #ClassesList

		local previousActionIndex = table.find(ClassesList, previousAction)

		local currentActionIndex = table.find(ClassesList, currentAction)

		local targetValue = rewardValue + (discountFactor * currentQVector[1][currentActionIndex] * (1 - terminalStateValue))

		local temporalDifferenceError = targetValue - previousQVector[1][previousActionIndex] 

		local outputDimensionSizeArray = {1, numberOfClasses}

		local temporalDifferenceErrorVector = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)

		temporalDifferenceErrorVector[1][previousActionIndex] = temporalDifferenceError

		if (EligibilityTrace) then

			EligibilityTrace:increment(1, previousActionIndex, discountFactor, outputDimensionSizeArray)

			temporalDifferenceErrorVector = EligibilityTrace:calculate(temporalDifferenceErrorVector)

		end
		
		local negatedTemporalDifferenceErrorVector = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorVector) -- The original non-deep SARSA version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error vector to make the neural network to perform gradient ascent.

		Model:forwardPropagate(previousFeatureVector, true)

		Model:update(negatedTemporalDifferenceErrorVector, true)
		
		local TargetModelParameters = Model:getModelParameters(true)

		TargetModelParameters = rateAverageModelParameters(NewDeepDoubleStateActionRewardStateActionModel.averagingRate, TargetModelParameters, PrimaryModelParameters)

		Model:setModelParameters(TargetModelParameters, true)
		
		return temporalDifferenceErrorVector

	end)
	
	NewDeepDoubleStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue) 
		
		local EligibilityTrace = NewDeepDoubleStateActionRewardStateActionModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	NewDeepDoubleStateActionRewardStateActionModel:setResetFunction(function()
		
		local EligibilityTrace = NewDeepDoubleStateActionRewardStateActionModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
	end)

	return NewDeepDoubleStateActionRewardStateActionModel

end

return DeepDoubleStateActionRewardStateActionModel
