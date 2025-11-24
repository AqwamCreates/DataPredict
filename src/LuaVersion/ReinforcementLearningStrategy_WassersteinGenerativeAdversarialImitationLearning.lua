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

local AqwamTensorLibrary = require("AqwamTensorLibrary")

local GenerativeAdversarialImitationLearningBaseModel = require("Model_GenerativeAdversarialImitationLearningBaseModel")

local WassersteinGenerativeAdversarialImitationLearning = {}

WassersteinGenerativeAdversarialImitationLearning.__index = WassersteinGenerativeAdversarialImitationLearning

setmetatable(WassersteinGenerativeAdversarialImitationLearning, GenerativeAdversarialImitationLearningBaseModel)

function WassersteinGenerativeAdversarialImitationLearning.new(parameterDictionary)
	
	local NewWassersteinGenerativeAdversarialImitationLearning = GenerativeAdversarialImitationLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewWassersteinGenerativeAdversarialImitationLearning, WassersteinGenerativeAdversarialImitationLearning)
	
	NewWassersteinGenerativeAdversarialImitationLearning:setName("WassersteinGenerativeAdversarialImitationLearning")
	
	NewWassersteinGenerativeAdversarialImitationLearning:setCategoricalTrainFunction(function(previousFeatureMatrix, expertPreviousActionMatrix, currentFeatureMatrix, expertCurrentActionMatrix, terminalStateMatrix)
		
		local DiscriminatorModel = NewWassersteinGenerativeAdversarialImitationLearning.DiscriminatorModel

		local ReinforcementLearningModel = NewWassersteinGenerativeAdversarialImitationLearning.ReinforcementLearningModel

		if (not DiscriminatorModel) then error("No discriminator neural network.") end

		if (not ReinforcementLearningModel) then error("No reinforcement learning neural network.") end

		local numberOfStepsPerEpisode = NewWassersteinGenerativeAdversarialImitationLearning.numberOfStepsPerEpisode

		local isOutputPrinted = NewWassersteinGenerativeAdversarialImitationLearning.isOutputPrinted

		local ActionsList = ReinforcementLearningModel:getActionsList()

		local previousFeatureMatrixTable = NewWassersteinGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(previousFeatureMatrix, numberOfStepsPerEpisode)

		local expertPreviousActionMatrixTable = NewWassersteinGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(expertPreviousActionMatrix, numberOfStepsPerEpisode)

		local currentFeatureMatrixTable = NewWassersteinGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(currentFeatureMatrix, numberOfStepsPerEpisode)
		
		local expertCurrentActionMatrixTable = NewWassersteinGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(expertCurrentActionMatrix, numberOfStepsPerEpisode)
		
		local terminalStateMatrixTable = NewWassersteinGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(terminalStateMatrix, numberOfStepsPerEpisode)

		local discriminatorInputNumberOfFeatures, discriminatorInputHasBias = DiscriminatorModel:getLayer(1)

		if (discriminatorInputNumberOfFeatures ~= (#expertPreviousActionMatrix[1] + #previousFeatureMatrix[1])) then error("The number of input neurons for the discriminator does not match the total number of both state features and expert actions.") end

		local discriminatorExpertLossGradientMatrix = {{1}}
		
		local costArray = {}

		local discriminatorLoss

		for episode = 1, #previousFeatureMatrixTable, 1 do

			local previousFeatureSubMatrix = previousFeatureMatrixTable[episode]

			local expertPreviousActionSubMatrix = expertPreviousActionMatrixTable[episode]

			local currentFeatureSubMatrix = currentFeatureMatrixTable[episode]
			
			local expertCurrentActionMatrix = expertCurrentActionMatrixTable[episode]
			
			local terminalStateSubMatrix = terminalStateMatrixTable[episode]

			for step = 1, numberOfStepsPerEpisode, 1 do

				task.wait()

				local previousFeatureVector = {previousFeatureSubMatrix[step]}

				local expertPreviousActionVector = {expertPreviousActionSubMatrix[step]}

				local currentFeatureVector = {currentFeatureSubMatrix[step]}
				
				local expertCurrentActionVector = {expertCurrentActionMatrix[step]}
				
				local terminalStateValue = terminalStateSubMatrix[step][1]

				local agentActionVector = ReinforcementLearningModel:predict(previousFeatureVector, true)

				local concatenatedExpertStateActionVector = AqwamTensorLibrary:concatenate(previousFeatureVector, expertPreviousActionVector, 2)

				local concatenatedAgentStateActionVector = AqwamTensorLibrary:concatenate(previousFeatureVector, agentActionVector, 2)

				if (discriminatorInputHasBias) then

					table.insert(concatenatedExpertStateActionVector[1], 1)

					table.insert(concatenatedAgentStateActionVector[1], 1)

				end

				local discriminatorExpertActionValueMatrix = DiscriminatorModel:forwardPropagate(concatenatedExpertStateActionVector, true)

				DiscriminatorModel:update(discriminatorExpertLossGradientMatrix, true)

				local discriminatorAgentActionValueMatrix = DiscriminatorModel:forwardPropagate(concatenatedAgentStateActionVector, true)

				local discriminatorAgentLossGradientMatrix = {{-discriminatorAgentActionValueMatrix[1][1]}}

				DiscriminatorModel:update(discriminatorAgentLossGradientMatrix, true)

				local expertPreviousActionIndex = NewWassersteinGenerativeAdversarialImitationLearning:chooseIndexWithHighestValue(expertPreviousActionVector)

				local expertCurrentActionIndex = NewWassersteinGenerativeAdversarialImitationLearning:chooseIndexWithHighestValue(expertCurrentActionVector)

				local expertPreviousAction = ActionsList[expertPreviousActionIndex]

				local expertCurrentAction = ActionsList[expertCurrentActionIndex]

				if (not expertPreviousAction) then error("Missing previous action at index " .. expertPreviousActionIndex .. ".") end

				if (not expertCurrentAction) then error("Missing current action at index " .. expertCurrentActionIndex .. ".") end

				discriminatorLoss = AqwamTensorLibrary:subtract(discriminatorExpertActionValueMatrix, discriminatorAgentActionValueMatrix)[1][1]

				ReinforcementLearningModel:categoricalUpdate(previousFeatureVector, expertPreviousAction, discriminatorLoss, currentFeatureVector, expertCurrentAction, terminalStateValue)

				table.insert(costArray, discriminatorLoss)

				if (isOutputPrinted) then print("Episode: " .. episode .. "\t\tStep: " .. step .. "\t\tDiscriminator Loss: " .. discriminatorLoss) end

			end

			ReinforcementLearningModel:episodeUpdate(1)

		end
		
		if (isOutputPrinted) then

			if (discriminatorLoss == math.huge) then warn("The model diverged.") end

			if (discriminatorLoss ~= discriminatorLoss) then warn("The model produced nan (not a number) values.") end

		end

		return costArray
		
	end)
	
	NewWassersteinGenerativeAdversarialImitationLearning:setDiagonalGaussianTrainFunction(function(previousFeatureMatrix, expertPreviousActionMeanMatrix, expertPreviousActionStandardDeviationMatrix, expertPreviousActionNoiseMatrix, currentFeatureMatrix, expertCurrentActionMeanMatrix, terminalStateMatrix)
		
		local DiscriminatorModel = NewWassersteinGenerativeAdversarialImitationLearning.DiscriminatorModel

		local ReinforcementLearningModel = NewWassersteinGenerativeAdversarialImitationLearning.ReinforcementLearningModel

		if (not DiscriminatorModel) then error("No discriminator neural network.") end

		if (not ReinforcementLearningModel) then error("No reinforcement learning neural network.") end

		local numberOfStepsPerEpisode = NewWassersteinGenerativeAdversarialImitationLearning.numberOfStepsPerEpisode

		local isOutputPrinted = NewWassersteinGenerativeAdversarialImitationLearning.isOutputPrinted

		local previousFeatureMatrixTable = NewWassersteinGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(previousFeatureMatrix, numberOfStepsPerEpisode)

		local expertPreviousActionMeanMatrixTable = NewWassersteinGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(expertPreviousActionMeanMatrix, numberOfStepsPerEpisode)

		local expertPreviousActionStandardDeviationTable = NewWassersteinGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(expertPreviousActionStandardDeviationMatrix, numberOfStepsPerEpisode)
		
		local expertPreviousActionNoiseTable = NewWassersteinGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(expertPreviousActionNoiseMatrix, numberOfStepsPerEpisode)

		local currentFeatureMatrixTable = NewWassersteinGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(currentFeatureMatrix, numberOfStepsPerEpisode)
		
		local expertCurrentActionMeanMatrixTable = NewWassersteinGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(expertCurrentActionMeanMatrix, numberOfStepsPerEpisode)
		
		local terminalStateMatrixTable = NewWassersteinGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(terminalStateMatrix, numberOfStepsPerEpisode)

		local discriminatorInputNumberOfFeatures, discriminatorInputHasBias = DiscriminatorModel:getLayer(1)

		if (discriminatorInputNumberOfFeatures ~= (#expertPreviousActionMeanMatrix[1] + #previousFeatureMatrix[1])) then error("The number of input neurons for the discriminator does not match the total number of both state features and expert actions.") end

		local discriminatorExpertLossGradientMatrix = {{1}}

		local costArray = {}
		
		local discriminatorLoss

		for episode = 1, #previousFeatureMatrixTable, 1 do

			local previousFeatureSubMatrix = previousFeatureMatrixTable[episode]

			local expertPreviousActionMeanSubMatrix = expertPreviousActionMeanMatrixTable[episode]

			local expertPreviousActionStandardDeviationSubMatrix = expertPreviousActionStandardDeviationTable[episode]
			
			local expertPreviousActionNoiseSubMatrix = expertPreviousActionNoiseTable[episode]

			local currentFeatureSubMatrix = currentFeatureMatrixTable[episode]
			
			local expertCurrentActionMeanSubMatrix = expertCurrentActionMeanMatrixTable[episode]
			
			local terminalStateSubMatrix = terminalStateMatrixTable[episode]

			for step = 1, numberOfStepsPerEpisode, 1 do

				task.wait()

				local previousFeatureVector = {previousFeatureSubMatrix[step]}

				local expertPreviousActionMeanVector = {expertPreviousActionMeanSubMatrix[step]}

				local expertPreviousActionStandardDeviationVector = {expertPreviousActionStandardDeviationSubMatrix[step]}
				
				local expertPreviousActionNoiseVector = {expertPreviousActionNoiseSubMatrix[step]}

				local currentFeatureVector = {currentFeatureSubMatrix[step]}
				
				local expertCurrentActionMeanVector = {expertCurrentActionMeanSubMatrix[step]}
				
				local terminalStateValue = terminalStateSubMatrix[step][1]

				local agentPreviousActionMeanVector = ReinforcementLearningModel:predict(previousFeatureVector, true)

				local concatenatedExpertStateActionVector = AqwamTensorLibrary:concatenate(previousFeatureVector, expertPreviousActionMeanVector, 2)

				local concatenatedAgentStateActionVector = AqwamTensorLibrary:concatenate(previousFeatureVector, agentPreviousActionMeanVector, 2)

				if (discriminatorInputHasBias) then

					table.insert(concatenatedExpertStateActionVector[1], 1)

					table.insert(concatenatedAgentStateActionVector[1], 1)

				end

				local discriminatorExpertActionValueMatrix = DiscriminatorModel:forwardPropagate(concatenatedExpertStateActionVector, true)

				DiscriminatorModel:backwardPropagate(discriminatorExpertLossGradientMatrix, true)

				local discriminatorAgentActionValueMatrix = DiscriminatorModel:forwardPropagate(concatenatedAgentStateActionVector, true)

				local discriminatorAgentLossGradientMatrix = {{-discriminatorAgentActionValueMatrix[1][1]}}

				DiscriminatorModel:backwardPropagate(discriminatorAgentLossGradientMatrix, true)

				discriminatorLoss = AqwamTensorLibrary:subtract(discriminatorExpertActionValueMatrix, discriminatorAgentActionValueMatrix)[1][1]

				ReinforcementLearningModel:diagonalGaussianUpdate(previousFeatureVector, expertPreviousActionMeanVector, expertPreviousActionStandardDeviationVector, expertPreviousActionNoiseVector, discriminatorLoss, currentFeatureVector, expertCurrentActionMeanVector, terminalStateValue)

				table.insert(costArray, discriminatorLoss)

				if (isOutputPrinted) then print("Episode: " .. episode .. "\t\tStep: " .. step .. "\t\tDiscriminator Loss: " .. discriminatorLoss) end

			end

			ReinforcementLearningModel:episodeUpdate(1)

		end
		
		if (isOutputPrinted) then

			if (discriminatorLoss == math.huge) then warn("The model diverged.") end

			if (discriminatorLoss ~= discriminatorLoss) then warn("The model produced nan (not a number) values.") end

		end

		return costArray
		
	end)
	
	return NewWassersteinGenerativeAdversarialImitationLearning
	
end

return WassersteinGenerativeAdversarialImitationLearning
