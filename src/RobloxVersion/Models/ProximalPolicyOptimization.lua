local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local ReinforcementLearningActorCriticBaseModel = require(script.Parent.ReinforcementLearningActorCriticBaseModel)

ProximalPolicyOptimizationModel = {}

ProximalPolicyOptimizationModel.__index = ProximalPolicyOptimizationModel

setmetatable(ProximalPolicyOptimizationModel, ReinforcementLearningActorCriticBaseModel)

local function calculateProbability(outputMatrix)

	local meanVector = AqwamMatrixLibrary:horizontalMean(outputMatrix)

	local standardDeviationVector = AqwamMatrixLibrary:horizontalStandardDeviation(outputMatrix)

	local zScoreVectorPart1 = AqwamMatrixLibrary:subtract(outputMatrix, meanVector)

	local zScoreVector = AqwamMatrixLibrary:divide(zScoreVectorPart1, standardDeviationVector)

	local zScoreSquaredVector = AqwamMatrixLibrary:power(zScoreVector, 2)

	local probabilityVectorPart1 = AqwamMatrixLibrary:multiply(-0.5, zScoreSquaredVector)

	local probabilityVectorPart2 = AqwamMatrixLibrary:multiply(standardDeviationVector, math.sqrt(2 * math.pi))

	local probabilityVector = AqwamMatrixLibrary:divide(probabilityVectorPart1, probabilityVectorPart2)

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

function ProximalPolicyOptimizationModel.new(discountFactor)
	
	local NewProximalPolicyOptimizationModel = ReinforcementLearningActorCriticBaseModel.new(discountFactor)
	
	setmetatable(NewProximalPolicyOptimizationModel, ProximalPolicyOptimizationModel)
	
	local rewardHistory = {}
	
	local criticValueHistory = {}
	
	local actionVectorHistory = {}
	
	local oldActionVectorHistory = {}
	
	local advantageValueHistory = {}
	
	local oldAdvantageValueHistory = {}
	
	NewProximalPolicyOptimizationModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local allOutputsMatrix = NewProximalPolicyOptimizationModel.ActorModel:predict(previousFeatureVector, true)

		local actionProbabilityVector = calculateProbability(allOutputsMatrix)
		
		local CriticModel = NewProximalPolicyOptimizationModel.CriticModel

		local previousCriticValue = CriticModel:predict(previousFeatureVector, true)[1][1]

		local currentCriticValue = CriticModel:predict(currentFeatureVector, true)[1][1]

		local advantageValue = rewardValue + (NewProximalPolicyOptimizationModel.discountFactor * currentCriticValue) - previousCriticValue

		table.insert(advantageValueHistory, advantageValue)

		table.insert(criticValueHistory, previousCriticValue)

		table.insert(actionVectorHistory, actionProbabilityVector)
		
		table.insert(rewardHistory, rewardValue)
		
	end)
	
	NewProximalPolicyOptimizationModel:setEpisodeUpdateFunction(function()
		
		if (#oldActionVectorHistory == 0) then 

			oldActionVectorHistory = table.clone(actionVectorHistory)
			
			oldAdvantageValueHistory = table.clone(advantageValueHistory)
			
			table.clear(advantageValueHistory)

			table.clear(criticValueHistory)

			table.clear(rewardHistory)

			table.clear(actionVectorHistory)

			return 

		end
		
		local rewardsToGoArray = calculateRewardsToGo(rewardHistory, NewProximalPolicyOptimizationModel.discountFactor)

		local sumActorLossVector = AqwamMatrixLibrary:createMatrix(1, #NewProximalPolicyOptimizationModel.ClassesList)

		local historyLength = #criticValueHistory

		local sumCriticLoss = 0

		for h = 1, historyLength, 1 do

			local currentActionVector = actionVectorHistory[h]

			local previousActionVector = oldActionVectorHistory[h]

			local ratioVector = AqwamMatrixLibrary:divide(currentActionVector, previousActionVector)

			local actorLossVector = AqwamMatrixLibrary:multiply(ratioVector, oldAdvantageValueHistory[h])

			local criticLoss = math.pow(rewardsToGoArray[h] - criticValueHistory[h], 2)

			sumActorLossVector = AqwamMatrixLibrary:add(sumActorLossVector, actorLossVector)

			sumCriticLoss += criticLoss

		end

		local calculatedActorLossVector = AqwamMatrixLibrary:divide(-sumActorLossVector, historyLength)

		local calculatedCriticLoss = sumCriticLoss / historyLength
		
		local ActorModel = NewProximalPolicyOptimizationModel.ActorModel
		
		local CriticModel = NewProximalPolicyOptimizationModel.CriticModel
		
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
	
	NewProximalPolicyOptimizationModel:extendResetFunction(function()
		
		table.clear(advantageValueHistory)
		
		table.clear(oldAdvantageValueHistory)
		
		table.clear(criticValueHistory)

		table.clear(rewardHistory)

		table.clear(actionVectorHistory)
		
		table.clear(oldActionVectorHistory)
		
	end)
	
	return NewProximalPolicyOptimizationModel
	
end

return ProximalPolicyOptimizationModel
