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

local DeepReinforcementLearningActorCriticBaseModel = require(script.Parent.DeepReinforcementLearningActorCriticBaseModel)

local VanillaPolicyGradientModel = {}

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

	local actionProbabilityGradientVectorHistory = {}

	local rewardValueHistory = {}

	local advantageValueHistory = {}

	NewVanillaPolicyGradientModel:setCategoricalUpdateFunction(function(previousFeatureVector, previousAction, rewardValue, currentFeatureVector, currentAction, terminalStateValue)

		local ActorModel = NewVanillaPolicyGradientModel.ActorModel

		local CriticModel = NewVanillaPolicyGradientModel.CriticModel

		local actionVector = ActorModel:forwardPropagate(previousFeatureVector)

		local actionProbabilityVector = calculateProbability(actionVector)

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureVector)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureVector)[1][1]

		local advantageValue = rewardValue + (NewVanillaPolicyGradientModel.discountFactor * currentCriticValue) - previousCriticValue

		local ClassesList = ActorModel:getClassesList()

		local classIndex = table.find(ClassesList, previousAction)

		local actionProbabilityGradientVector = {}

		for i, _ in ipairs(ClassesList) do

			actionProbabilityGradientVector[i] = (((i == classIndex) and 1) or 0) - actionProbabilityVector[1][i]

		end

		actionProbabilityGradientVector = {actionProbabilityGradientVector}

		table.insert(featureVectorHistory, previousFeatureVector)

		table.insert(actionProbabilityGradientVectorHistory, actionProbabilityGradientVector)

		table.insert(rewardValueHistory, rewardValue)

		table.insert(advantageValueHistory, advantageValue)

		return advantageValue

	end)

	NewVanillaPolicyGradientModel:setDiagonalGaussianUpdateFunction(function(previousFeatureVector, previousActionMeanVector, previousActionStandardDeviationVector, previousActionNoiseVector, rewardValue, currentFeatureVector, currentActionMeanVector, terminalStateValue)

		if (not previousActionNoiseVector) then previousActionNoiseVector = AqwamTensorLibrary:createRandomNormalTensor({1, #previousActionMeanVector[1]}) end

		local CriticModel = NewVanillaPolicyGradientModel.CriticModel

		local actionVectorPart1 = AqwamTensorLibrary:multiply(previousActionStandardDeviationVector, previousActionNoiseVector)

		local actionVector = AqwamTensorLibrary:add(previousActionMeanVector, actionVectorPart1)

		local actionProbabilityGradientVectorPart1 = AqwamTensorLibrary:subtract(actionVector, previousActionMeanVector)

		local actionProbabilityGradientVectorPart2 = AqwamTensorLibrary:power(previousActionStandardDeviationVector, 2)

		local actionProbabilityGradientVector = AqwamTensorLibrary:divide(actionProbabilityGradientVectorPart1, actionProbabilityGradientVectorPart2)

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureVector)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureVector)[1][1]

		local advantageValue = rewardValue + (NewVanillaPolicyGradientModel.discountFactor * currentCriticValue) - previousCriticValue

		table.insert(featureVectorHistory, previousFeatureVector)

		table.insert(actionProbabilityGradientVectorHistory, actionProbabilityGradientVector)

		table.insert(rewardValueHistory, rewardValue)

		table.insert(advantageValueHistory, advantageValue)

		return advantageValue

	end)

	NewVanillaPolicyGradientModel:setEpisodeUpdateFunction(function(terminalStateValue)

		local ActorModel = NewVanillaPolicyGradientModel.ActorModel

		local CriticModel = NewVanillaPolicyGradientModel.CriticModel

		for h, featureVector in ipairs(featureVectorHistory) do

			local advantageValue = advantageValueHistory[h]

			local actorLossVector = AqwamTensorLibrary:multiply(actionProbabilityGradientVectorHistory[h], advantageValue)

			actorLossVector = AqwamTensorLibrary:unaryMinus(actorLossVector)

			CriticModel:forwardPropagate(featureVector, true)

			ActorModel:forwardPropagate(featureVector, true)

			CriticModel:update(advantageValue, true)

			ActorModel:update(actorLossVector, true)

		end

		table.clear(featureVectorHistory)

		table.clear(actionProbabilityGradientVectorHistory)

		table.clear(rewardValueHistory)

		table.clear(advantageValueHistory)

	end)

	NewVanillaPolicyGradientModel:setResetFunction(function()

		table.clear(featureVectorHistory)

		table.clear(actionProbabilityGradientVectorHistory)

		table.clear(rewardValueHistory)

		table.clear(advantageValueHistory)

	end)

	return NewVanillaPolicyGradientModel

end

return VanillaPolicyGradientModel
