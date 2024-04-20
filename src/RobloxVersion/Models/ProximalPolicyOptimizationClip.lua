local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local ReinforcementLearningActorCriticBaseModel = require(script.Parent.ReinforcementLearningActorCriticBaseModel)

ProximalPolicyOptimizationClipModel = {}

ProximalPolicyOptimizationClipModel.__index = ProximalPolicyOptimizationClipModel

setmetatable(ProximalPolicyOptimizationClipModel, ReinforcementLearningActorCriticBaseModel)

local defaultClipRatio = 0.3

local function calculateProbability(outputMatrix)

	local meanVector = AqwamMatrixLibrary:horizontalMean(outputMatrix)

	local standardDeviationVector = AqwamMatrixLibrary:horizontalStandardDeviation(outputMatrix)

	local zScoreVectorPart1 = AqwamMatrixLibrary:subtract(outputMatrix, meanVector)

	local zScoreVector = AqwamMatrixLibrary:divide(zScoreVectorPart1, standardDeviationVector)

	local zScoreSquaredVector = AqwamMatrixLibrary:power(zScoreVector, 2)

	local probabilityVectorPart1 = AqwamMatrixLibrary:multiply(-0.5, zScoreSquaredVector)

	local probabilityVectorPart2 = AqwamMatrixLibrary:applyFunction(math.exp, probabilityVectorPart1)

	local probabilityVectorPart3 = AqwamMatrixLibrary:multiply(standardDeviationVector, math.sqrt(2 * math.pi))

	local probabilityVector = AqwamMatrixLibrary:divide(probabilityVectorPart2, probabilityVectorPart3)

	return probabilityVector

end

local function calculateRewardsToGo(rewardHistory, discountFactor)

	local rewardsToGoArray = {}

	local discountedReward = 0

	for h = #rewardHistory, 1, -1 do

		discountedReward = rewardHistory[h] + (discountFactor * discountedReward)

		table.insert(rewardsToGoArray, 1, discountedReward)

	end

	return rewardsToGoArray

end

function ProximalPolicyOptimizationClipModel.new(clipRatio, discountFactor)
	
	local NewProximalPolicyOptimizationClipModel = ReinforcementLearningActorCriticBaseModel.new(discountFactor)
	
	setmetatable(NewProximalPolicyOptimizationClipModel, ProximalPolicyOptimizationClipModel)
	
	NewProximalPolicyOptimizationClipModel.clipRatio = clipRatio or defaultClipRatio
	
	local rewardHistory = {}
	
	local criticValueHistory = {}
	
	local actionVectorHistory = {}
	
	local oldActionVectorHistory = {}
	
	local advantageValueHistory = {}
	
	local oldAdvantageValueHistory = {}
	
	NewProximalPolicyOptimizationClipModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local allOutputsMatrix = NewProximalPolicyOptimizationClipModel.ActorModel:predict(previousFeatureVector, true)

		local actionProbabilityVector = calculateProbability(allOutputsMatrix)
		
		local CriticModel = NewProximalPolicyOptimizationClipModel.CriticModel

		local previousCriticValue = CriticModel:predict(previousFeatureVector, true)[1][1]

		local currentCriticValue = CriticModel:predict(currentFeatureVector, true)[1][1]

		local advantageValue = rewardValue + (NewProximalPolicyOptimizationClipModel.discountFactor * currentCriticValue) - previousCriticValue

		table.insert(advantageValueHistory, advantageValue)

		table.insert(criticValueHistory, previousCriticValue)

		table.insert(actionVectorHistory, actionProbabilityVector)
		
		table.insert(rewardHistory, rewardValue)
		
	end)
	
	NewProximalPolicyOptimizationClipModel:setEpisodeUpdateFunction(function()
		
		if (#oldActionVectorHistory == 0) then 
			
			oldActionVectorHistory = table.clone(actionVectorHistory)
			
			oldAdvantageValueHistory = table.clone(advantageValueHistory)
			
			table.clear(advantageValueHistory)

			table.clear(criticValueHistory)

			table.clear(rewardHistory)

			table.clear(actionVectorHistory)
			
			return 
				
		end
		
		local rewardsToGoArray = calculateRewardsToGo(rewardHistory, NewProximalPolicyOptimizationClipModel.discountFactor)

		local sumActorLossVector = AqwamMatrixLibrary:createMatrix(1, #NewProximalPolicyOptimizationClipModel.ClassesList)

		local historyLength = #criticValueHistory

		local sumCriticLoss = 0
		
		local clipFunction = function(value) 
			
			local clipRatio = NewProximalPolicyOptimizationClipModel.clipRatio 
			
			return math.clamp(value, 1 - clipRatio, 1 + clipRatio) 
			
		end

		for h = 1, historyLength, 1 do

			local currentActionVector = actionVectorHistory[h]

			local previousActionVector = oldActionVectorHistory[h]

			local ratioVector = AqwamMatrixLibrary:divide(currentActionVector, previousActionVector)
			
			local advantageValue = advantageValueHistory[h]
			
			local surrogateLoss1 = AqwamMatrixLibrary:multiply(ratioVector, advantageValue)
			
			local surrogateLoss2Part1 = AqwamMatrixLibrary:applyFunction(clipFunction, ratioVector)
			
			local surrogateLoss2 = AqwamMatrixLibrary:multiply(surrogateLoss2Part1, advantageValue)

			local actorLossVector = AqwamMatrixLibrary:applyFunction(math.min, surrogateLoss1, surrogateLoss2)

			local criticLoss = math.pow(rewardsToGoArray[h] - criticValueHistory[h], 2)

			sumActorLossVector = AqwamMatrixLibrary:add(sumActorLossVector, actorLossVector)

			sumCriticLoss += criticLoss

		end

		local calculatedActorLossVector = AqwamMatrixLibrary:divide(-sumActorLossVector, historyLength)

		local calculatedCriticLoss = sumCriticLoss / historyLength
		
		local ActorModel = NewProximalPolicyOptimizationClipModel.ActorModel
		
		local CriticModel = NewProximalPolicyOptimizationClipModel.CriticModel
		
		local numberOfFeatures, hasBias = ActorModel:getLayer(1)

		numberOfFeatures += (hasBias and 1) or 0

		local featureVector = AqwamMatrixLibrary:createMatrix(1, numberOfFeatures, 1)

		ActorModel:forwardPropagate(featureVector, true)
		CriticModel:forwardPropagate(featureVector, true)

		ActorModel:backPropagate(calculatedActorLossVector, true)
		CriticModel:backPropagate(calculatedCriticLoss, true)
		
		oldActionVectorHistory = table.clone(actionVectorHistory)
		
		oldAdvantageValueHistory = table.clone(advantageValueHistory)
		
		table.clear(advantageValueHistory)

		table.clear(criticValueHistory)

		table.clear(rewardHistory)

		table.clear(actionVectorHistory)
		
	end)
	
	NewProximalPolicyOptimizationClipModel:extendResetFunction(function()
		
		table.clear(advantageValueHistory)
		
		table.clear(oldAdvantageValueHistory)

		table.clear(criticValueHistory)

		table.clear(rewardHistory)

		table.clear(actionVectorHistory)
		
		table.clear(oldActionVectorHistory)
		
	end)
	
	return NewProximalPolicyOptimizationClipModel
	
end

function ProximalPolicyOptimizationClipModel:setParameters(clipRatio, discountFactor)
	
	self.clipRatio = clipRatio or self.clipRatio

	self.discountFactor =  discountFactor or self.discountFactor
	
end

return ProximalPolicyOptimizationClipModel
