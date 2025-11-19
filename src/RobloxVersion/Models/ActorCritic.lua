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

ActorCriticModel = {}

ActorCriticModel.__index = ActorCriticModel

setmetatable(ActorCriticModel, DeepReinforcementLearningActorCriticBaseModel)

local function calculateProbability(valueVector)

	local maximumValue = AqwamTensorLibrary:findMaximumValue(valueVector)

	local zValueVector = AqwamTensorLibrary:subtract(valueVector, maximumValue)

	local exponentVector = AqwamTensorLibrary:exponent(zValueVector)

	local sumExponentValue = AqwamTensorLibrary:sum(exponentVector)

	local probabilityVector = AqwamTensorLibrary:divide(exponentVector, sumExponentValue)

	return probabilityVector

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

function ActorCriticModel.new(parameterDictionary)

	local NewActorCriticModel = DeepReinforcementLearningActorCriticBaseModel.new(parameterDictionary)

	setmetatable(NewActorCriticModel, ActorCriticModel)

	NewActorCriticModel:setName("ActorCritic")

	local featureVectorHistory = {}

	local actionProbabilityGradientVectorHistory = {}

	local rewardValueHistory = {}

	local criticValueHistory = {}

	NewActorCriticModel:setCategoricalUpdateFunction(function(previousFeatureVector, previousAction, rewardValue, currentFeatureVector, currentAction, terminalStateValue)
		
		local ActorModel = NewActorCriticModel.ActorModel
		
		local actionVector = ActorModel:forwardPropagate(previousFeatureVector, true)

		local criticValue = NewActorCriticModel.CriticModel:forwardPropagate(previousFeatureVector)[1][1]

		local actionProbabilityVector = calculateProbability(actionVector)

		local ClassesList = ActorModel:getClassesList()

		local classIndex = table.find(ClassesList, previousAction)
		
		local actionProbabilityGradientVector = {}
		
		for i, _ in ipairs(ClassesList) do
			
			actionProbabilityGradientVector[i] = ((i == classIndex and 1 or 0)) - actionProbabilityVector[1][i]
			
		end
		
		actionProbabilityGradientVector = {actionProbabilityGradientVector}

		table.insert(featureVectorHistory, previousFeatureVector)

		table.insert(actionProbabilityGradientVectorHistory, actionProbabilityGradientVector)

		table.insert(rewardValueHistory, rewardValue)

		table.insert(criticValueHistory, criticValue)

	end)

	NewActorCriticModel:setDiagonalGaussianUpdateFunction(function(previousFeatureVector, previousActionMeanVector, previousActionStandardDeviationVector, previousActionNoiseVector, rewardValue, currentFeatureVector, currentActionMeanVector, terminalStateValue)

		if (not previousActionNoiseVector) then previousActionNoiseVector = AqwamTensorLibrary:createRandomNormalTensor({1, #previousActionMeanVector[1]}) end

		local actionVectorPart1 = AqwamTensorLibrary:multiply(previousActionStandardDeviationVector, previousActionNoiseVector)

		local actionVector = AqwamTensorLibrary:add(previousActionMeanVector, actionVectorPart1)

		local actionProbabilityGradientVectorPart1 = AqwamTensorLibrary:subtract(actionVector, previousActionMeanVector)
		
		local actionProbabilityGradientVectorPart2 = AqwamTensorLibrary:power(previousActionStandardDeviationVector, 2)

		local actionProbabilityGradientVector = AqwamTensorLibrary:divide(actionProbabilityGradientVectorPart1, actionProbabilityGradientVectorPart2)

		local criticValue = NewActorCriticModel.CriticModel:forwardPropagate(previousFeatureVector)[1][1]

		table.insert(featureVectorHistory, previousFeatureVector)

		table.insert(actionProbabilityGradientVectorHistory, actionProbabilityGradientVector)

		table.insert(rewardValueHistory, rewardValue)

		table.insert(criticValueHistory, criticValue)

	end)

	NewActorCriticModel:setEpisodeUpdateFunction(function(terminalStateValue)

		local ActorModel = NewActorCriticModel.ActorModel

		local CriticModel = NewActorCriticModel.CriticModel

		local rewardToGoHistory = calculateRewardToGo(rewardValueHistory, NewActorCriticModel.discountFactor)

		for h, featureVector in ipairs(featureVectorHistory) do

			local criticLoss = rewardToGoHistory[h] - criticValueHistory[h]

			local actorLossVector = AqwamTensorLibrary:multiply(actionProbabilityGradientVectorHistory[h], criticLoss)

			actorLossVector = AqwamTensorLibrary:unaryMinus(actorLossVector)

			CriticModel:forwardPropagate(featureVector, true)

			ActorModel:forwardPropagate(featureVector, true)

			CriticModel:update(criticLoss, true)

			ActorModel:update(actorLossVector, true)

		end

		table.clear(featureVectorHistory)

		table.clear(actionProbabilityGradientVectorHistory)

		table.clear(rewardValueHistory)

		table.clear(criticValueHistory)

	end)

	NewActorCriticModel:setResetFunction(function()

		table.clear(featureVectorHistory)

		table.clear(actionProbabilityGradientVectorHistory)

		table.clear(rewardValueHistory)

		table.clear(criticValueHistory)

	end)

	return NewActorCriticModel

end

return ActorCriticModel
