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

local ReinforcementLearningNeuralNetworkBaseModel = require(script.Parent.ReinforcementLearningNeuralNetworkBaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

StateActionRewardStateActionNeuralNetworkModel = {}

StateActionRewardStateActionNeuralNetworkModel.__index = StateActionRewardStateActionNeuralNetworkModel

setmetatable(StateActionRewardStateActionNeuralNetworkModel, ReinforcementLearningNeuralNetworkBaseModel)

function StateActionRewardStateActionNeuralNetworkModel.new(maxNumberOfIterations, learningRate, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)

	local NewStateActionRewardStateActionNeuralNetworkModel = ReinforcementLearningNeuralNetworkBaseModel.new(maxNumberOfIterations, learningRate, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)

	setmetatable(NewStateActionRewardStateActionNeuralNetworkModel, StateActionRewardStateActionNeuralNetworkModel)

	NewStateActionRewardStateActionNeuralNetworkModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)

		local targetVector = NewStateActionRewardStateActionNeuralNetworkModel:predict(currentFeatureVector, true)

		local dicountedVector = AqwamMatrixLibrary:multiply(NewStateActionRewardStateActionNeuralNetworkModel.discountFactor, targetVector)

		local newTargetVector = AqwamMatrixLibrary:add(rewardValue, dicountedVector)

		NewStateActionRewardStateActionNeuralNetworkModel:train(previousFeatureVector, newTargetVector)
		
		return newTargetVector

	end)

	return NewStateActionRewardStateActionNeuralNetworkModel

end

function StateActionRewardStateActionNeuralNetworkModel:setParameters(maxNumberOfIterations, learningRate, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or self.numberOfReinforcementsPerEpisode

	self.epsilon = epsilon or self.epsilon

	self.epsilonDecayFactor =  epsilonDecayFactor or self.epsilonDecayFactor

	self.discountFactor =  discountFactor or self.discountFactor

	self.currentEpsilon = epsilon or self.currentEpsilon

end

return StateActionRewardStateActionNeuralNetworkModel
