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

AdvantageActorCriticModel = {}

AdvantageActorCriticModel.__index = AdvantageActorCriticModel

setmetatable(AdvantageActorCriticModel, DeepReinforcementLearningActorCriticBaseModel)

local defaultLambda = 0

local function calculateProbability(valueVector)

	local maximumValue = AqwamTensorLibrary:findMaximumValue(valueVector)

	local zValueVector = AqwamTensorLibrary:subtract(valueVector, maximumValue)

	local exponentVector = AqwamTensorLibrary:exponent(zValueVector)

	local sumExponentValue = AqwamTensorLibrary:sum(exponentVector)

	local probabilityVector = AqwamTensorLibrary:divide(exponentVector, sumExponentValue)

	return probabilityVector

end

function AdvantageActorCriticModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewAdvantageActorCriticModel = DeepReinforcementLearningActorCriticBaseModel.new(parameterDictionary)

	setmetatable(NewAdvantageActorCriticModel, AdvantageActorCriticModel)

	AdvantageActorCriticModel:setName("AdvantageActorCritic")

	NewAdvantageActorCriticModel.lambda = parameterDictionary.lambda or defaultLambda

	local featureVectorHistory = {}

	local advantageValueHistory = {}

	local actionProbabilityGradientVectorHistory = {}

	NewAdvantageActorCriticModel:setCategoricalUpdateFunction(function(previousFeatureVector, previousAction, rewardValue, currentFeatureVector, currentAction, terminalStateValue)
		
		local ActorModel = NewAdvantageActorCriticModel.ActorModel
		
		local CriticModel = NewAdvantageActorCriticModel.CriticModel

		local actionVector = ActorModel:forwardPropagate(previousFeatureVector)

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureVector)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureVector)[1][1]

		local actionProbabilityVector = calculateProbability(actionVector)

		local advantageValue = rewardValue + (NewAdvantageActorCriticModel.discountFactor * (1 - terminalStateValue) * currentCriticValue) - previousCriticValue

		local ClassesList = ActorModel:getClassesList()

		local classIndex = table.find(ClassesList, previousAction)

		local actionProbabilityGradientVector = {}

		for i, _ in ipairs(ClassesList) do

			actionProbabilityGradientVector[i] = (((i == classIndex) and 1) or 0) - actionProbabilityVector[1][i]

		end

		actionProbabilityGradientVector = {actionProbabilityGradientVector}

		table.insert(featureVectorHistory, previousFeatureVector)

		table.insert(actionProbabilityGradientVectorHistory, actionProbabilityGradientVector)

		table.insert(advantageValueHistory, advantageValue)

		return advantageValue

	end)

	NewAdvantageActorCriticModel:setDiagonalGaussianUpdateFunction(function(previousFeatureVector, previousActionMeanVector, previousActionStandardDeviationVector, previousActionNoiseVector, rewardValue, currentFeatureVector, currentActionMeanVector, terminalStateValue)

		if (not previousActionNoiseVector) then previousActionNoiseVector = AqwamTensorLibrary:createRandomNormalTensor({1, #previousActionMeanVector[1]}) end

		local CriticModel = NewAdvantageActorCriticModel.CriticModel

		local actionVectorPart1 = AqwamTensorLibrary:multiply(previousActionStandardDeviationVector, previousActionNoiseVector)

		local actionVector = AqwamTensorLibrary:add(previousActionMeanVector, actionVectorPart1)

		local actionProbabilityGradientVectorPart1 = AqwamTensorLibrary:subtract(actionVector, previousActionMeanVector)

		local actionProbabilityGradientVectorPart2 = AqwamTensorLibrary:power(previousActionStandardDeviationVector, 2)

		local actionProbabilityGradientVector = AqwamTensorLibrary:divide(actionProbabilityGradientVectorPart1, actionProbabilityGradientVectorPart2)

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureVector)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureVector)[1][1]

		local advantageValue = rewardValue + (NewAdvantageActorCriticModel.discountFactor * (1 - terminalStateValue) * currentCriticValue) - previousCriticValue

		table.insert(featureVectorHistory, previousFeatureVector)

		table.insert(actionProbabilityGradientVectorHistory, actionProbabilityGradientVector)

		table.insert(advantageValueHistory, advantageValue)

		return advantageValue

	end)

	NewAdvantageActorCriticModel:setEpisodeUpdateFunction(function()

		local ActorModel = NewAdvantageActorCriticModel.ActorModel

		local CriticModel = NewAdvantageActorCriticModel.CriticModel

		local lambda = NewAdvantageActorCriticModel.lambda

		if (lambda ~= 0) then

			local generalizedAdvantageEstimationValue = 0

			local generalizedAdvantageEstimationHistory = {}

			local discountFactor = NewAdvantageActorCriticModel.discountFactor

			for t = #advantageValueHistory, 1, -1 do

				generalizedAdvantageEstimationValue = advantageValueHistory[t] + (discountFactor * lambda * generalizedAdvantageEstimationValue)

				table.insert(generalizedAdvantageEstimationHistory, 1, generalizedAdvantageEstimationValue)

			end

			advantageValueHistory = generalizedAdvantageEstimationHistory

		end

		for h, featureVector in ipairs(featureVectorHistory) do

			local advantageValue = advantageValueHistory[h]
			
			advantageValue = -advantageValue

			local actorLossVector = AqwamTensorLibrary:multiply(actionProbabilityGradientVectorHistory[h], advantageValue)

			CriticModel:forwardPropagate(featureVector, true)

			ActorModel:forwardPropagate(featureVector, true)

			CriticModel:update(advantageValue, true)

			ActorModel:update(actorLossVector, true)

		end

		table.clear(featureVectorHistory)

		table.clear(actionProbabilityGradientVectorHistory)

		table.clear(advantageValueHistory)

	end)

	NewAdvantageActorCriticModel:setResetFunction(function()

		table.clear(featureVectorHistory)

		table.clear(actionProbabilityGradientVectorHistory)

		table.clear(advantageValueHistory)

	end)

	return NewAdvantageActorCriticModel

end

return AdvantageActorCriticModel
