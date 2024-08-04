--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

local ReinforcementLearningActorCriticBaseModel = require("Model_ReinforcementLearningActorCriticBaseModel")

VanillaPolicyGradientModel = {}

VanillaPolicyGradientModel.__index = VanillaPolicyGradientModel

setmetatable(VanillaPolicyGradientModel, ReinforcementLearningActorCriticBaseModel)

local function calculateRewardsToGo(rewardHistory, discountFactor)

	local rewardsToGoArray = {}

	local discountedReward = 0

	for h = #rewardHistory, 1, -1 do

		discountedReward = rewardHistory[h] + (discountFactor * discountedReward)

		table.insert(rewardsToGoArray, 1, discountedReward)

	end

	return rewardsToGoArray

end

function VanillaPolicyGradientModel.new(discountFactor)
	
	local NewVanillaPolicyGradientModel = ReinforcementLearningActorCriticBaseModel.new(discountFactor)

	setmetatable(NewVanillaPolicyGradientModel, VanillaPolicyGradientModel)

	local rewardHistory = {}

	local gradientHistory = {}
	
	local valueHistory = {}

	NewVanillaPolicyGradientModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)

		local allOutputsMatrix = NewVanillaPolicyGradientModel.ActorModel:predict(previousFeatureVector, true)

		local logOutputMatrix = AqwamMatrixLibrary:applyFunction(math.log, allOutputsMatrix)
		
		local CriticModel = NewVanillaPolicyGradientModel.CriticModel

		local previousCriticValue = CriticModel:predict(previousFeatureVector, true)[1][1]

		local currentCriticValue = CriticModel:predict(currentFeatureVector, true)[1][1]

		local advantageValue = rewardValue + (NewVanillaPolicyGradientModel.discountFactor * currentCriticValue) - previousCriticValue

		local gradientMatrix = AqwamMatrixLibrary:multiply(logOutputMatrix, advantageValue)
		
		table.insert(rewardHistory, rewardValue)
		
		table.insert(valueHistory, previousCriticValue)

		table.insert(gradientHistory, gradientMatrix[1])
		
		return advantageValue

	end)

	NewVanillaPolicyGradientModel:setEpisodeUpdateFunction(function()
		
		local episodeLength = #rewardHistory
		
		local rewardToGoArray = calculateRewardsToGo(rewardHistory, NewVanillaPolicyGradientModel.discountFactor)

		local sumGradient = AqwamMatrixLibrary:verticalSum(gradientHistory)
		
		local sumActorLossVector = AqwamMatrixLibrary:multiply(-1, sumGradient)
		
		local sumCriticLoss = 0
		
		for i, value in ipairs(valueHistory) do
			
			local criticLoss = value - rewardToGoArray[i]
			
			sumCriticLoss = sumCriticLoss + criticLoss
			
		end
		
		sumCriticLoss = sumCriticLoss / episodeLength
		
		local ActorModel = NewVanillaPolicyGradientModel.ActorModel

		local CriticModel = NewVanillaPolicyGradientModel.CriticModel

		local numberOfFeatures = ActorModel:getTotalNumberOfNeurons(1)

		local numberOfLayers = ActorModel:getNumberOfLayers()

		local numberOfNeuronsAtFinalLayer = ActorModel:getTotalNumberOfNeurons(numberOfLayers)

		local featureVector = AqwamMatrixLibrary:createMatrix(1, numberOfFeatures, 1)

		ActorModel:forwardPropagate(featureVector, true)
		CriticModel:forwardPropagate(featureVector, true)

		ActorModel:backPropagate(sumActorLossVector, true)
		CriticModel:backPropagate(sumCriticLoss, true)
		
		table.clear(rewardHistory)
		
		table.clear(valueHistory)
		
		table.clear(gradientHistory)

	end)

	NewVanillaPolicyGradientModel:extendResetFunction(function()

		table.clear(rewardHistory)
		
		table.clear(valueHistory)
		
		table.clear(gradientHistory)

	end)
	
	return NewVanillaPolicyGradientModel
	
end

return VanillaPolicyGradientModel