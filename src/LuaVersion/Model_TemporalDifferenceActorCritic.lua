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

local AqwamTensorLibrary = require("AqwamTensorLibrary")

local DeepReinforcementLearningActorCriticBaseModel = require("Model_DeepReinforcementLearningActorCriticBaseModel")

local TemporalDifferenceActorCriticModel = {}

TemporalDifferenceActorCriticModel.__index = TemporalDifferenceActorCriticModel

setmetatable(TemporalDifferenceActorCriticModel, DeepReinforcementLearningActorCriticBaseModel)

local defaultLambda = 0

local function calculateProbability(valueVector)

	local maximumValue = AqwamTensorLibrary:findMaximumValue(valueVector)

	local zValueVector = AqwamTensorLibrary:subtract(valueVector, maximumValue)

	local exponentVector = AqwamTensorLibrary:exponent(zValueVector)

	local sumExponentValue = AqwamTensorLibrary:sum(exponentVector)

	local probabilityVector = AqwamTensorLibrary:divide(exponentVector, sumExponentValue)

	return probabilityVector

end

function TemporalDifferenceActorCriticModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewTemporalDifferenceActorCriticModel = DeepReinforcementLearningActorCriticBaseModel.new(parameterDictionary)

	setmetatable(NewTemporalDifferenceActorCriticModel, TemporalDifferenceActorCriticModel)

	TemporalDifferenceActorCriticModel:setName("TemporalDifferenceActorCritic")

	NewTemporalDifferenceActorCriticModel.EligibilityTrace = parameterDictionary.EligibilityTrace

	NewTemporalDifferenceActorCriticModel:setCategoricalUpdateFunction(function(previousFeatureVector, previousAction, rewardValue, currentFeatureVector, currentAction, terminalStateValue)
		
		local ActorModel = NewTemporalDifferenceActorCriticModel.ActorModel
		
		local CriticModel = NewTemporalDifferenceActorCriticModel.CriticModel
		
		local discountFactor = NewTemporalDifferenceActorCriticModel.discountFactor

		local EligibilityTrace = NewTemporalDifferenceActorCriticModel.EligibilityTrace

		local actionVector = ActorModel:forwardPropagate(previousFeatureVector)

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureVector)[1][1]
		
		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureVector, true)[1][1]

		local actionProbabilityVector = calculateProbability(actionVector)

		local temporalDifferenceError = rewardValue + (discountFactor * (1 - terminalStateValue) * currentCriticValue) - previousCriticValue

		local ClassesList = ActorModel:getClassesList()
		
		local numberOfClasses = #ClassesList

		local classIndex = table.find(ClassesList, previousAction)

		local actionProbabilityGradientVector = {}

		for i, _ in ipairs(ClassesList) do

			actionProbabilityGradientVector[i] = (((i == classIndex) and 1) or 0) - actionProbabilityVector[1][i]

		end

		actionProbabilityGradientVector = {actionProbabilityGradientVector}

		if (EligibilityTrace) then
			
			local outputDimensionSizeArray = {1, numberOfClasses}

			local temporalDifferenceErrorVector = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0)

			temporalDifferenceErrorVector[1][classIndex] = temporalDifferenceError

			EligibilityTrace:increment(1, classIndex, discountFactor, outputDimensionSizeArray)

			temporalDifferenceErrorVector = EligibilityTrace:calculate(temporalDifferenceErrorVector)
			
			temporalDifferenceError = temporalDifferenceErrorVector[1][classIndex]

		end
		
		local criticLoss = -temporalDifferenceError

		local actorLossVector = AqwamTensorLibrary:multiply(criticLoss, actionProbabilityGradientVector)

		ActorModel:backwardPropagate(actorLossVector, true)

		CriticModel:backwardPropagate(criticLoss, true)

		return temporalDifferenceError

	end)

	NewTemporalDifferenceActorCriticModel:setDiagonalGaussianUpdateFunction(function(previousFeatureVector, previousActionMeanVector, previousActionStandardDeviationVector, previousActionNoiseVector, rewardValue, currentFeatureVector, currentActionMeanVector, terminalStateValue)

		if (not previousActionNoiseVector) then previousActionNoiseVector = AqwamTensorLibrary:createRandomNormalTensor({1, #previousActionMeanVector[1]}) end
		
		local ActorModel = NewTemporalDifferenceActorCriticModel.ActorModel
		
		local CriticModel = NewTemporalDifferenceActorCriticModel.CriticModel

		local actionVectorPart1 = AqwamTensorLibrary:multiply(previousActionStandardDeviationVector, previousActionNoiseVector)

		local actionVector = AqwamTensorLibrary:add(previousActionMeanVector, actionVectorPart1)

		local actionProbabilityGradientVectorPart1 = AqwamTensorLibrary:subtract(actionVector, previousActionMeanVector)

		local actionProbabilityGradientVectorPart2 = AqwamTensorLibrary:power(previousActionStandardDeviationVector, 2)

		local actionProbabilityGradientVector = AqwamTensorLibrary:divide(actionProbabilityGradientVectorPart1, actionProbabilityGradientVectorPart2)

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureVector)[1][1]
		
		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureVector, true)[1][1]

		local temporalDifferenceError = rewardValue + (NewTemporalDifferenceActorCriticModel.discountFactor * (1 - terminalStateValue) * currentCriticValue) - previousCriticValue
		
		local criticLoss = -temporalDifferenceError
		
		local actorLossVector = AqwamTensorLibrary:multiply(criticLoss, actionProbabilityGradientVector)
		
		ActorModel:forwardPropagate(previousFeatureVector, true)
		
		ActorModel:backwardPropagate(actorLossVector, true)
		
		CriticModel:backwardPropagate(criticLoss, true)
		
		return temporalDifferenceError

	end)

	NewTemporalDifferenceActorCriticModel:setEpisodeUpdateFunction(function()
		
		local EligibilityTrace = NewTemporalDifferenceActorCriticModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end

	end)

	NewTemporalDifferenceActorCriticModel:setResetFunction(function()
		
		local EligibilityTrace = NewTemporalDifferenceActorCriticModel.EligibilityTrace

		if (EligibilityTrace) then EligibilityTrace:reset() end
		
		
	end)

	return NewTemporalDifferenceActorCriticModel

end

return TemporalDifferenceActorCriticModel
