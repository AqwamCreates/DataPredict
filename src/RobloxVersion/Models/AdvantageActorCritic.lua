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

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local ReinforcementLearningActorCriticBaseModel = require(script.Parent.ReinforcementLearningActorCriticBaseModel)

AdvantageActorCriticModel = {}

AdvantageActorCriticModel.__index = AdvantageActorCriticModel

setmetatable(AdvantageActorCriticModel, ReinforcementLearningActorCriticBaseModel)

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

	local NewAdvantageActorCriticModel = ReinforcementLearningActorCriticBaseModel.new(parameterDictionary)

	setmetatable(NewAdvantageActorCriticModel, AdvantageActorCriticModel)
	
	NewAdvantageActorCriticModel:setName("AdvantageActorCritic")
	
	NewAdvantageActorCriticModel.lambda = parameterDictionary.lambda or defaultLambda
	
	local advantageValueHistory = {}

	local actionProbabilityVectorHistory = {}
	
	NewAdvantageActorCriticModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)
		
		local CriticModel = NewAdvantageActorCriticModel.CriticModel
		
		local actionVector = NewAdvantageActorCriticModel.ActorModel:forwardPropagate(previousFeatureVector)
		
		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureVector)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureVector)[1][1]

		local actionProbabilityVector = calculateProbability(actionVector)
		
		local advantageValue = rewardValue + (NewAdvantageActorCriticModel.discountFactor * (1 - terminalStateValue) * currentCriticValue) - previousCriticValue

		local logActionProbabilityVector = AqwamTensorLibrary:logarithm(actionProbabilityVector)
		
		table.insert(actionProbabilityVectorHistory, logActionProbabilityVector)

		table.insert(advantageValueHistory, advantageValue)

		return advantageValue

	end)
	
	NewAdvantageActorCriticModel:setDiagonalGaussianUpdateFunction(function(previousFeatureVector, actionMeanVector, actionStandardDeviationVector, actionNoiseVector, rewardValue, currentFeatureVector, terminalStateValue)

		if (not actionNoiseVector) then actionNoiseVector = AqwamTensorLibrary:createRandomUniformTensor({1, #actionMeanVector[1]}) end
		
		local CriticModel = NewAdvantageActorCriticModel.CriticModel

		local actionVectorPart1 = AqwamTensorLibrary:multiply(actionStandardDeviationVector, actionNoiseVector)

		local actionVector = AqwamTensorLibrary:add(actionMeanVector, actionVectorPart1)

		local zScoreVectorPart1 = AqwamTensorLibrary:subtract(actionVector, actionMeanVector)

		local zScoreVector = AqwamTensorLibrary:divide(zScoreVectorPart1, actionStandardDeviationVector)

		local squaredZScoreVector = AqwamTensorLibrary:power(zScoreVector, 2)

		local logActionProbabilityVectorPart1 = AqwamTensorLibrary:logarithm(actionStandardDeviationVector)

		local logActionProbabilityVectorPart2 = AqwamTensorLibrary:multiply(2, logActionProbabilityVectorPart1)

		local logActionProbabilityVectorPart3 = AqwamTensorLibrary:add(squaredZScoreVector, logActionProbabilityVectorPart2)

		local logActionProbabilityVectorPart4 = AqwamTensorLibrary:add(logActionProbabilityVectorPart3, math.log(2 * math.pi))
		
		local logActionProbabilityVector = AqwamTensorLibrary:multiply(-0.5, logActionProbabilityVectorPart4)

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureVector)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureVector)[1][1]

		local advantageValue = rewardValue + (NewAdvantageActorCriticModel.discountFactor * (1 - terminalStateValue) * currentCriticValue) - previousCriticValue
		
		table.insert(actionProbabilityVectorHistory, logActionProbabilityVector)

		table.insert(advantageValueHistory, advantageValue)
		
		return advantageValue

	end)

	NewAdvantageActorCriticModel:setEpisodeUpdateFunction(function(terminalStateValue)

		local sumActorLossVector = AqwamTensorLibrary:createTensor({1, #actionProbabilityVectorHistory[1]}, 0)
		
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
		
		local sumCriticLoss = 0
		
		for h, advantageValue in ipairs(advantageValueHistory) do
			
			local actorLossVector = AqwamTensorLibrary:multiply(actionProbabilityVectorHistory[h], advantageValue)

			sumCriticLoss = sumCriticLoss + advantageValue

			sumActorLossVector = AqwamTensorLibrary:add(sumActorLossVector, actorLossVector)
			
		end
		
		local ActorModel = NewAdvantageActorCriticModel.ActorModel

		local CriticModel = NewAdvantageActorCriticModel.CriticModel

		local numberOfFeatures = ActorModel:getTotalNumberOfNeurons(1)

		local featureVector = AqwamTensorLibrary:createTensor({1, numberOfFeatures}, 1)
		
		sumActorLossVector = AqwamTensorLibrary:unaryMinus(sumActorLossVector)

		ActorModel:forwardPropagate(featureVector, true, true)
		
		CriticModel:forwardPropagate(featureVector, true, true)

		ActorModel:backwardPropagate(sumActorLossVector, true)
		
		CriticModel:backwardPropagate(sumCriticLoss, true)
		
		table.clear(actionProbabilityVectorHistory)

		table.clear(advantageValueHistory)

	end)

	NewAdvantageActorCriticModel:setResetFunction(function()
		
		table.clear(actionProbabilityVectorHistory)

		table.clear(advantageValueHistory)

	end)
	
	return NewAdvantageActorCriticModel

end

return AdvantageActorCriticModel