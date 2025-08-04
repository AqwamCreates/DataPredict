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

local AqwamTensorLibrary = require("AqwamTensorLibrary")

local DeepReinforcementLearningActorCriticBaseModel = require("Model_DeepReinforcementLearningActorCriticBaseModel")

VanillaPolicyGradientModel = {}

VanillaPolicyGradientModel.__index = VanillaPolicyGradientModel

setmetatable(VanillaPolicyGradientModel, DeepReinforcementLearningActorCriticBaseModel)

local function calculateProbability(valueVector)

	local highestActionValue = AqwamTensorLibrary:findMaximumValue(valueVector)

	local subtractedZVector = AqwamTensorLibrary:subtract(valueVector, highestActionValue)

	local exponentActionVector = AqwamTensorLibrary:applyFunction(math.exp, subtractedZVector)

	local exponentActionSumVector = AqwamTensorLibrary:sum(exponentActionVector, 2)

	local targetActionVector = AqwamTensorLibrary:divide(exponentActionVector, exponentActionSumVector)

	return targetActionVector

end

local function calculateRewardToGo(rewardHistory, discountFactor)

	local rewardToGoArray = {}

	local discountedReward = 0

	for h = #rewardHistory, 1, -1 do

		discountedReward = rewardHistory[h] + (discountFactor * discountedReward)

		table.insert(rewardToGoArray, 1, discountedReward)

	end

	return rewardToGoArray

end

function VanillaPolicyGradientModel.new(parameterDictionary)

	local NewVanillaPolicyGradientModel = DeepReinforcementLearningActorCriticBaseModel.new(parameterDictionary)

	setmetatable(NewVanillaPolicyGradientModel, VanillaPolicyGradientModel)

	NewVanillaPolicyGradientModel:setName("VanillaPolicyGradient")

	local featureVectorHistory = {}

	local actionProbabilityVectorHistory = {}

	local rewardValueHistory = {}

	local advantageValueHistory = {}

	NewVanillaPolicyGradientModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)

		local CriticModel = NewVanillaPolicyGradientModel.CriticModel

		local actionVector = NewVanillaPolicyGradientModel.ActorModel:forwardPropagate(previousFeatureVector)

		local actionProbabilityVector = calculateProbability(actionVector)

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureVector)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureVector)[1][1]

		local advantageValue = rewardValue + (NewVanillaPolicyGradientModel.discountFactor * currentCriticValue) - previousCriticValue

		local logActionProbabilityVector = AqwamTensorLibrary:logarithm(actionProbabilityVector)

		local actorLossVector = AqwamTensorLibrary:multiply(logActionProbabilityVector, advantageValue)

		table.insert(featureVectorHistory, previousFeatureVector)

		table.insert(actionProbabilityVectorHistory, logActionProbabilityVector)

		table.insert(rewardValueHistory, rewardValue)

		table.insert(advantageValueHistory, advantageValue)

		return advantageValue

	end)

	NewVanillaPolicyGradientModel:setDiagonalGaussianUpdateFunction(function(previousFeatureVector, actionMeanVector, actionStandardDeviationVector, actionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue)
		
		if (not actionNoiseVector) then actionNoiseVector = AqwamTensorLibrary:createRandomNormalTensor({1, #actionMeanVector[1]}) end

		local CriticModel = NewVanillaPolicyGradientModel.CriticModel

		local actionVectorPart1 = AqwamTensorLibrary:multiply(actionStandardDeviationVector, actionNoiseVector)

		local actionVector = AqwamTensorLibrary:add(actionMeanVector, actionVectorPart1)

		local zScoreVectorPart1 = AqwamTensorLibrary:subtract(actionVector, actionMeanVector)

		local zScoreVector = AqwamTensorLibrary:divide(zScoreVectorPart1, actionStandardDeviationVector)

		local squaredZScoreVector = AqwamTensorLibrary:power(zScoreVector, 2)

		local logActionProbabilityVectorPart1 = AqwamTensorLibrary:logarithm(actionStandardDeviationVector)

		local logActionProbabilityVectorPart2 = AqwamTensorLibrary:multiply(2, logActionProbabilityVectorPart1)

		local logActionProbabilityVectorPart3 = AqwamTensorLibrary:add(squaredZScoreVector, logActionProbabilityVectorPart2)

		local logActionProbabilityVector = AqwamTensorLibrary:add(logActionProbabilityVectorPart3, math.log(2 * math.pi))

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureVector)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureVector)[1][1]

		local advantageValue = rewardValue + (NewVanillaPolicyGradientModel.discountFactor * currentCriticValue) - previousCriticValue

		local actorLossVector = AqwamTensorLibrary:multiply(logActionProbabilityVector, advantageValue)

		table.insert(featureVectorHistory, previousFeatureVector)

		table.insert(actionProbabilityVectorHistory, logActionProbabilityVector)

		table.insert(rewardValueHistory, rewardValue)

		table.insert(advantageValueHistory, advantageValue)

		return advantageValue

	end)

	NewVanillaPolicyGradientModel:setEpisodeUpdateFunction(function(terminalStateValue)

		local ActorModel = NewVanillaPolicyGradientModel.ActorModel

		local CriticModel = NewVanillaPolicyGradientModel.CriticModel

		for h, featureVector in ipairs(featureVectorHistory) do

			local advantageValue = advantageValueHistory[h]

			local actorLossVector = AqwamTensorLibrary:multiply(actionProbabilityVectorHistory[h], advantageValue)

			actorLossVector = AqwamTensorLibrary:unaryMinus(actorLossVector)

			CriticModel:forwardPropagate(featureVector, true)

			ActorModel:forwardPropagate(featureVector, true)

			CriticModel:update(advantageValue, true)

			ActorModel:update(actorLossVector, true)

		end

		table.clear(featureVectorHistory)

		table.clear(actionProbabilityVectorHistory)

		table.clear(rewardValueHistory)

		table.clear(advantageValueHistory)

	end)

	NewVanillaPolicyGradientModel:setResetFunction(function()

		table.clear(featureVectorHistory)

		table.clear(actionProbabilityVectorHistory)

		table.clear(rewardValueHistory)

		table.clear(advantageValueHistory)

	end)

	return NewVanillaPolicyGradientModel

end

return VanillaPolicyGradientModel
