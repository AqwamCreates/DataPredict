--[[

	--------------------------------------------------------------------

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
	
	DO NOT SELL, RENT, DISTRIBUTE THIS LIBRARY
	
	DO NOT SELL, RENT, DISTRIBUTE MODIFIED VERSION OF THIS LIBRARY
	
	DO NOT CLAIM OWNERSHIP OF THIS LIBRARY
	
	GIVE CREDIT AND SOURCE WHEN USING THIS LIBRARY IF YOUR USAGE FALLS UNDER ONE OF THESE CATEGORIES:
	
		- USED AS A VIDEO OR ARTICLE CONTENT
		- USED AS RESEARCH AND EDUCATION CONTENT
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

local ReinforcementLearningNeuralNetworkBaseModel = require("Model_ReinforcementLearningNeuralNetworkBaseModel")

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

REINFORCENeuralNetworkModel = {}

REINFORCENeuralNetworkModel.__index = REINFORCENeuralNetworkModel

setmetatable(REINFORCENeuralNetworkModel, ReinforcementLearningNeuralNetworkBaseModel)

local function calculateRewardsToGo(rewardHistory, discountFactor)

	local rewardsToGoArray = {}

	local discountedReward = 0

	for h = #rewardHistory, 1, -1 do

		discountedReward += rewardHistory[h] + (discountFactor * discountedReward)

		table.insert(rewardsToGoArray, 1, discountedReward)

	end

	return rewardsToGoArray

end

function REINFORCENeuralNetworkModel.new(maxNumberOfIterations, learningRate, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)

	local NewREINFORCENeuralNetworkModel = ReinforcementLearningNeuralNetworkBaseModel.new(maxNumberOfIterations, learningRate, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)
	
	setmetatable(NewREINFORCENeuralNetworkModel, REINFORCENeuralNetworkModel)
	
	local targetVectorArray = {}
	
	local rewardArray = {}
	
	NewREINFORCENeuralNetworkModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)

		local predictedVector = NewREINFORCENeuralNetworkModel:predict(previousFeatureVector, true)
		
		local logPredictedVector = AqwamMatrixLibrary:applyFunction(math.log, predictedVector)
		
		local targetVector = AqwamMatrixLibrary:multiply(logPredictedVector, rewardValue)

		table.insert(targetVectorArray, targetVector)
		
		table.insert(rewardArray, rewardValue)

	end)
	
	NewREINFORCENeuralNetworkModel:setEpisodeUpdateFunction(function()
		
		local rewardsToGoArray = calculateRewardsToGo(rewardArray, discountFactor)
		
		local lossVector = AqwamMatrixLibrary:createMatrix(1, #NewREINFORCENeuralNetworkModel.ClassesList)
		
		for i = 1, #targetVectorArray, 1 do
			
			local discountedReward = AqwamMatrixLibrary:multiply(targetVectorArray[i], rewardsToGoArray[i])
			
			lossVector = AqwamMatrixLibrary:add(lossVector, discountedReward)
			
		end
		
		local numberOfNeurons = NewREINFORCENeuralNetworkModel.numberOfNeuronsTable[1] + NewREINFORCENeuralNetworkModel.hasBiasNeuronTable[1]

		local inputVector = {table.create(numberOfNeurons, 1)}
		
		NewREINFORCENeuralNetworkModel:forwardPropagate(inputVector, true)

		NewREINFORCENeuralNetworkModel:backPropagate(lossVector, true)
		
		table.clear(targetVectorArray)
		table.clear(rewardArray)
		
	end)
	
	NewREINFORCENeuralNetworkModel:extendResetFunction(function()

		table.clear(targetVectorArray)
		table.clear(rewardArray)
		
	end)

	return NewREINFORCENeuralNetworkModel

end

function REINFORCENeuralNetworkModel:setParameters(maxNumberOfIterations, learningRate, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)
	
	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate
	
	self.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or self.numberOfReinforcementsPerEpisode

	self.epsilon = epsilon or self.epsilon

	self.epsilonDecayFactor =  epsilonDecayFactor or self.epsilonDecayFactor

	self.discountFactor =  discountFactor or self.discountFactor

	self.currentEpsilon = epsilon or self.currentEpsilon

end

return REINFORCENeuralNetworkModel
