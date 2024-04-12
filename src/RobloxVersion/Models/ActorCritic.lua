local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local ReinforcementLearningActorCriticBaseModel = require(script.Parent.ReinforcementLearningActorCriticBaseModel)

ActorCriticModel = {}

ActorCriticModel.__index = ActorCriticModel

setmetatable(ActorCriticModel, ReinforcementLearningActorCriticBaseModel)

local function sampleAction(actionProbabilityVector)

	local totalProbability = 0

	for _, probability in ipairs(actionProbabilityVector[1]) do

		totalProbability += probability

	end

	local randomValue = math.random() * totalProbability

	local cumulativeProbability = 0

	local actionIndex = 1

	for i, probability in ipairs(actionProbabilityVector[1]) do

		cumulativeProbability += probability

		if (randomValue > cumulativeProbability) then continue end

		actionIndex = i

		break

	end

	return actionIndex

end

local function calculateProbability(outputMatrix)

	local sumVector = AqwamMatrixLibrary:horizontalSum(outputMatrix)

	local result = AqwamMatrixLibrary:divide(outputMatrix, sumVector)

	return result

end

function ActorCriticModel.new(discountFactor)
	
	local NewActorCriticModel = ReinforcementLearningActorCriticBaseModel.new(discountFactor)
	
	setmetatable(NewActorCriticModel, ActorCriticModel)
	
	local actionProbabilityHistory = {}
	
	local criticValueHistory = {}
	
	local rewardHistory = {}
	
	NewActorCriticModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local allOutputsMatrix = NewActorCriticModel.ActorModel:predict(previousFeatureVector, true)

		local actionProbabilityVector = calculateProbability(allOutputsMatrix)

		local criticValue = NewActorCriticModel.CriticModel:predict(previousFeatureVector, true)[1][1]

		local numberOfActions = #allOutputsMatrix[1]

		local actionIndex = sampleAction(actionProbabilityVector)

		local actionProbability = actionProbabilityVector[1][actionIndex]

		table.insert(actionProbabilityHistory, actionProbability)

		table.insert(criticValueHistory, criticValue)

		table.insert(rewardHistory, rewardValue)
		
	end)
	
	NewActorCriticModel:setEpisodeUpdateFunction(function()
		
		local returnsHistory = {}

		local discountedSum = 0

		local historyLength = #rewardHistory

		for h = historyLength, 1, -1 do

			discountedSum = rewardHistory[h] + NewActorCriticModel.discountFactor * discountedSum

			table.insert(returnsHistory, 1, discountedSum)

		end

		local sumActorLosses = 0

		local sumCriticLosses = 0

		for h = 1, historyLength, 1 do

			local criticValue = criticValueHistory[h]

			local returns = returnsHistory[h]

			local actionProbability = actionProbabilityHistory[h]

			local actorLoss = math.log(actionProbability) * (returns - criticValue) 

			local criticLoss = (returns - criticValue)^2

			sumActorLosses += actorLoss

			sumCriticLosses += criticLoss

		end
		
		local ActorModel = NewActorCriticModel.ActorModel

		local CriticModel = NewActorCriticModel.CriticModel
		
		local numberOfFeatures, hasBias = ActorModel:getLayer(1)
		
		numberOfFeatures += (hasBias and 1) or 0

		local featureVector = AqwamMatrixLibrary:createMatrix(1, numberOfFeatures, 1)
		local lossVector = AqwamMatrixLibrary:createMatrix(1, #NewActorCriticModel.ClassesList, -sumActorLosses)

		ActorModel:forwardPropagate(featureVector, true)
		CriticModel:forwardPropagate(featureVector, true)

		ActorModel:backPropagate(lossVector, true)
		CriticModel:backPropagate(-sumCriticLosses, true)

		table.clear(actionProbabilityHistory)

		table.clear(criticValueHistory)

		table.clear(rewardHistory)
		
	end)
	
	NewActorCriticModel:extendResetFunction(function()
		
		table.clear(actionProbabilityHistory)

		table.clear(criticValueHistory)

		table.clear(rewardHistory)
		
	end)
	
	return NewActorCriticModel
	
end

return ActorCriticModel
