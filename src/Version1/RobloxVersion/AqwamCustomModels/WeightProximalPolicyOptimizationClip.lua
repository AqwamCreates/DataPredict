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

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local ReinforcementLearningActorCriticBaseModel = require(script.Parent.Parent.Models.ReinforcementLearningActorCriticBaseModel)

WeightProximalPolicyOptimizationClipModel = {}

WeightProximalPolicyOptimizationClipModel.__index = WeightProximalPolicyOptimizationClipModel

setmetatable(WeightProximalPolicyOptimizationClipModel, ReinforcementLearningActorCriticBaseModel)

local defaultClipRatio = 0.3

local function calculateRewardToGo(rewardHistory, discountFactor)

	local rewardToGoArray = {}

	local discountedReward = 0

	for h = #rewardHistory, 1, -1 do

		discountedReward = rewardHistory[h] + (discountFactor * discountedReward)

		table.insert(rewardToGoArray, 1, discountedReward)

	end

	return rewardToGoArray

end

function WeightProximalPolicyOptimizationClipModel.new(clipRatio, discountFactor)

	local NewWeightProximalPolicyOptimizationClipModel = ReinforcementLearningActorCriticBaseModel.new(discountFactor)

	setmetatable(NewWeightProximalPolicyOptimizationClipModel, WeightProximalPolicyOptimizationClipModel)

	NewWeightProximalPolicyOptimizationClipModel.clipRatio = clipRatio or defaultClipRatio

	local rewardValueHistory = {}

	local criticValueHistory = {}

	local advantageValueHistory = {}

	local oldAdvantageValueHistory = {}

	local OldActorModelParameters

	local OldCriticModelParameters

	NewWeightProximalPolicyOptimizationClipModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)

		local CriticModel = NewWeightProximalPolicyOptimizationClipModel.CriticModel

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureVector)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureVector)[1][1]

		local advantageValue = rewardValue + (NewWeightProximalPolicyOptimizationClipModel.discountFactor * currentCriticValue) - previousCriticValue

		table.insert(advantageValueHistory, advantageValue)

		table.insert(criticValueHistory, currentCriticValue)

		table.insert(rewardValueHistory, rewardValue)

		return advantageValue

	end)

	NewWeightProximalPolicyOptimizationClipModel:setDiagonalGaussianUpdateFunction(function(previousFeatureVector, actionMeanVector, actionStandardDeviationVector, rewardValue, currentFeatureVector)

		local CriticModel = NewWeightProximalPolicyOptimizationClipModel.CriticModel

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureVector)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureVector)[1][1]

		local advantageValue = rewardValue + (NewWeightProximalPolicyOptimizationClipModel.discountFactor * currentCriticValue) - previousCriticValue

		table.insert(advantageValueHistory, advantageValue)

		table.insert(criticValueHistory, currentCriticValue)

		table.insert(rewardValueHistory, rewardValue)

		return advantageValue

	end)

	NewWeightProximalPolicyOptimizationClipModel:setEpisodeUpdateFunction(function()

		local ActorModel = NewWeightProximalPolicyOptimizationClipModel.ActorModel

		local CriticModel = NewWeightProximalPolicyOptimizationClipModel.CriticModel

		if (OldActorModelParameters == nil)  or (OldCriticModelParameters == nil) then 

			OldActorModelParameters = ActorModel:getModelParameters(false)

			OldCriticModelParameters = CriticModel:getModelParameters(false)

			ActorModel:setModelParameters(nil)

			CriticModel:setModelParameters(nil)

			oldAdvantageValueHistory = table.clone(advantageValueHistory)

			table.clear(advantageValueHistory)

			table.clear(criticValueHistory)

			table.clear(rewardValueHistory)

			return 

		end

		local rewardToGoArray = calculateRewardToGo(rewardValueHistory, NewWeightProximalPolicyOptimizationClipModel.discountFactor)

		local historyLength = #criticValueHistory

		local sumAdvantageValue = 0

		local sumCriticLoss = 0

		local clipFunction = function(value) 

			local clipRatio = NewWeightProximalPolicyOptimizationClipModel.clipRatio 

			return math.clamp(value, 1 - clipRatio, 1 + clipRatio) 

		end

		for h = 1, historyLength, 1 do

			local advantageValue = advantageValueHistory[h]

			local criticLoss = math.pow(rewardToGoArray[h] - criticValueHistory[h], 2)

			sumAdvantageValue = sumAdvantageValue + advantageValue

			sumCriticLoss = sumCriticLoss + criticLoss

		end

		local NewActorModelParameters = {}

		local NewCriticModelParameters = {}

		for i, weightMatrix in ipairs(ActorModel:getModelParameters(false)) do
			
			local oldWeightMatrix = OldActorModelParameters[i]

			local weightRatio = AqwamMatrixLibrary:divide(weightMatrix, oldWeightMatrix)

			weightRatio = AqwamMatrixLibrary:applyFunction(clipFunction, weightRatio)
			
			local advantageWeightRatioMatrix = AqwamMatrixLibrary:multiply(weightRatio, sumAdvantageValue)

			NewActorModelParameters[i] = AqwamMatrixLibrary:multiply(oldWeightMatrix, weightRatio, sumAdvantageValue)

		end

		for i, weightMatrix in ipairs(CriticModel:getModelParameters(false)) do
			
			local oldWeightMatrix = OldActorModelParameters[i]

			local weightRatio = AqwamMatrixLibrary:divide(weightMatrix, OldActorModelParameters[i])

			weightRatio = AqwamMatrixLibrary:applyFunction(clipFunction, weightRatio)

			NewCriticModelParameters[i] = AqwamMatrixLibrary:multiply(oldWeightMatrix, weightRatio, sumCriticLoss)

		end

		ActorModel:setModelParameters(NewActorModelParameters)

		CriticModel:setModelParameters(NewCriticModelParameters)

		oldAdvantageValueHistory = table.clone(advantageValueHistory)

		table.clear(advantageValueHistory)

		table.clear(criticValueHistory)

		table.clear(rewardValueHistory)

	end)

	NewWeightProximalPolicyOptimizationClipModel:setResetFunction(function()

		table.clear(advantageValueHistory)

		table.clear(oldAdvantageValueHistory)

		table.clear(criticValueHistory)

		table.clear(rewardValueHistory)

	end)

	return NewWeightProximalPolicyOptimizationClipModel

end

function WeightProximalPolicyOptimizationClipModel:setParameters(clipRatio, discountFactor)

	self.clipRatio = clipRatio or self.clipRatio

	self.discountFactor =  discountFactor or self.discountFactor

end

return WeightProximalPolicyOptimizationClipModel