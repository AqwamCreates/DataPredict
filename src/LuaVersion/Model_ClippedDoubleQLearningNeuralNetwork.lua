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

ClippedDoubleQLearningNeuralNetworkModel = {}

ClippedDoubleQLearningNeuralNetworkModel.__index = ClippedDoubleQLearningNeuralNetworkModel

setmetatable(ClippedDoubleQLearningNeuralNetworkModel, ReinforcementLearningNeuralNetworkBaseModel)

function ClippedDoubleQLearningNeuralNetworkModel.new(maxNumberOfIterations, learningRate, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)

	local NewClippedDoubleQLearningNeuralNetworkModel = ReinforcementLearningNeuralNetworkBaseModel.new(maxNumberOfIterations, learningRate, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)

	NewClippedDoubleQLearningNeuralNetworkModel.ModelParametersArray = {}
	
	NewClippedDoubleQLearningNeuralNetworkModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)

		local maxQValues = {}

		for i = 1, 2, 1 do

			NewClippedDoubleQLearningNeuralNetworkModel:setModelParameters(NewClippedDoubleQLearningNeuralNetworkModel.ModelParametersArray[i], true)

			local predictedValue, maxQValue = NewClippedDoubleQLearningNeuralNetworkModel:predict(currentFeatureVector)

			table.insert(maxQValues, maxQValue[1][1])

		end

		local maxQValue = math.min(table.unpack(maxQValues))

		local target = rewardValue + (NewClippedDoubleQLearningNeuralNetworkModel.discountFactor * maxQValue)

		local actionIndex = table.find(NewClippedDoubleQLearningNeuralNetworkModel.ClassesList, action)

		for i = 1, 2, 1 do

			NewClippedDoubleQLearningNeuralNetworkModel:setModelParameters(NewClippedDoubleQLearningNeuralNetworkModel.ModelParametersArray[i], true)

			local targetVector = NewClippedDoubleQLearningNeuralNetworkModel:predict(previousFeatureVector, true)

			targetVector[1][actionIndex] = maxQValue

			NewClippedDoubleQLearningNeuralNetworkModel:train(previousFeatureVector, targetVector)

		end

	end)

	return NewClippedDoubleQLearningNeuralNetworkModel

end

function ClippedDoubleQLearningNeuralNetworkModel:setParameters(maxNumberOfIterations, learningRate, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or self.numberOfReinforcementsPerEpisode

	self.epsilon = epsilon or self.epsilon

	self.epsilonDecayFactor =  epsilonDecayFactor or self.epsilonDecayFactor

	self.discountFactor =  discountFactor or self.discountFactor

	self.currentEpsilon = epsilon or self.currentEpsilon

end

function ClippedDoubleQLearningNeuralNetworkModel:setModelParameters1(ModelParameters1)

	self.ModelParametersArray[1] = ModelParameters1

end

function ClippedDoubleQLearningNeuralNetworkModel:setModelParameters2(ModelParameters2)

	self.ModelParametersArray[2] = ModelParameters2

end

function ClippedDoubleQLearningNeuralNetworkModel:getModelParameters1(ModelParameters1)

	return self.ModelParametersArray[1]

end

function ClippedDoubleQLearningNeuralNetworkModel:getModelParameters2(ModelParameters2)

	return self.ModelParametersArray[2]

end

return ClippedDoubleQLearningNeuralNetworkModel
