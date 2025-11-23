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

local AqwamTensorLibraryLinker = require("AqwamTensorLibrary")

local GenerativeAdversarialImitationLearningBaseModel = require("ReinforcementLearningStrategy_GenerativeAdversarialImitationLearningBaseModel")

local GenerativeAdversarialImitationLearning = {}

GenerativeAdversarialImitationLearning.__index = GenerativeAdversarialImitationLearning

setmetatable(GenerativeAdversarialImitationLearning, GenerativeAdversarialImitationLearningBaseModel)

function GenerativeAdversarialImitationLearning.new(parameterDictionary)
	
	local NewGenerativeAdversarialImitationLearning = GenerativeAdversarialImitationLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewGenerativeAdversarialImitationLearning, GenerativeAdversarialImitationLearning)
	
	NewGenerativeAdversarialImitationLearning:setName("GenerativeAdversarialImitationLearning")
	
	NewGenerativeAdversarialImitationLearning:setCategoricalTrainFunction(function(previousFeatureMatrix, expertPreviousActionMatrix, currentFeatureMatrix, expertCurrentActionMatrix, terminalStateMatrix)
		
		local DiscriminatorModel = NewGenerativeAdversarialImitationLearning.DiscriminatorModel

		local ReinforcementLearningModel = NewGenerativeAdversarialImitationLearning.ReinforcementLearningModel

		if (not DiscriminatorModel) then error("No discriminator neural network.") end

		if (not ReinforcementLearningModel) then error("No reinforcement learning neural network.") end

		local numberOfStepsPerEpisode = NewGenerativeAdversarialImitationLearning.numberOfStepsPerEpisode

		local isOutputPrinted = NewGenerativeAdversarialImitationLearning.isOutputPrinted

		local ActionsList = ReinforcementLearningModel:getActionsList()

		local previousFeatureMatrixTable = NewGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(previousFeatureMatrix, numberOfStepsPerEpisode)

		local expertPreviousActionMatrixTable = NewGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(expertPreviousActionMatrix, numberOfStepsPerEpisode)

		local currentFeatureMatrixTable = NewGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(currentFeatureMatrix, numberOfStepsPerEpisode)
		
		local expertCurrentActionMatrixTable = NewGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(expertCurrentActionMatrix, numberOfStepsPerEpisode)
		
		local terminalStateMatrixTable = NewGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(terminalStateMatrix, numberOfStepsPerEpisode)

		local discriminatorInputNumberOfFeatures, discriminatorInputHasBias = DiscriminatorModel:getLayer(1)

		if (discriminatorInputNumberOfFeatures ~= (#expertPreviousActionMatrix[1] + #previousFeatureMatrix[1])) then error("The number of input neurons for the discriminator does not match the total number of both state features and expert actions.") end

		discriminatorInputNumberOfFeatures = discriminatorInputNumberOfFeatures + ((discriminatorInputHasBias and 1) or 0)

		local discriminatorInputVector = AqwamTensorLibraryLinker:createTensor({1, discriminatorInputNumberOfFeatures}, 1)

		for episode = 1, #previousFeatureMatrixTable, 1 do

			local previousFeatureSubMatrix = previousFeatureMatrixTable[episode]

			local expertPreviousActionSubMatrix = expertPreviousActionMatrixTable[episode]

			local currentFeatureSubMatrix = currentFeatureMatrixTable[episode]
			
			local expertCurrentActionSubMatrix = expertCurrentActionMatrixTable[episode]
			
			local terminalStateSubMatrix = terminalStateMatrixTable[episode]

			for step = 1, numberOfStepsPerEpisode, 1 do

				task.wait()

				local previousFeatureVector = {previousFeatureSubMatrix[step]}

				local expertPreviousActionVector = {expertPreviousActionSubMatrix[step]}

				local currentFeatureVector = {currentFeatureSubMatrix[step]}
				
				local expertCurrentActionVector = {expertCurrentActionSubMatrix[step]}
				
				local terminalStateValue = terminalStateSubMatrix[step][1]

				local agentActionVector = ReinforcementLearningModel:predict(previousFeatureVector, true)

				local concatenatedExpertStateActionVector = AqwamTensorLibraryLinker:concatenate(previousFeatureVector, expertPreviousActionVector, 2)

				local concatenatedAgentStateActionVector = AqwamTensorLibraryLinker:concatenate(previousFeatureVector, agentActionVector, 2)

				if (discriminatorInputHasBias) then

					table.insert(concatenatedExpertStateActionVector[1], 1)

					table.insert(concatenatedAgentStateActionVector[1], 1)

				end

				local discriminatorExpertActionValue = DiscriminatorModel:predict(concatenatedExpertStateActionVector, true)[1][1]

				local discriminatorAgentActionValue = DiscriminatorModel:predict(concatenatedAgentStateActionVector, true)[1][1]

				local discriminatorLoss = math.log(discriminatorExpertActionValue) + math.log(1 - discriminatorAgentActionValue)

				local expertPreviousActionIndex = NewGenerativeAdversarialImitationLearning:chooseIndexWithHighestValue(expertPreviousActionVector)

				local expertCurrentActionIndex = NewGenerativeAdversarialImitationLearning:chooseIndexWithHighestValue(expertCurrentActionVector)

				local expertPreviousAction = ActionsList[expertPreviousActionIndex]

				local expertCurrentAction = ActionsList[expertCurrentActionIndex]

				if (not expertPreviousAction) then error("Missing previous action at index " .. expertPreviousActionIndex .. ".") end

				if (not expertCurrentAction) then error("Missing current action at index " .. expertCurrentActionIndex .. ".") end

				ReinforcementLearningModel:categoricalUpdate(previousFeatureVector, expertPreviousAction, discriminatorLoss, currentFeatureVector, expertCurrentAction, terminalStateValue)

				DiscriminatorModel:forwardPropagate(discriminatorInputVector, true)

				DiscriminatorModel:backwardPropagate(discriminatorLoss, true)

				if (isOutputPrinted) then print("Episode: " .. episode .. "\t\tStep: " .. step .. "\t\tDiscriminator Loss: " .. discriminatorLoss) end

			end

			ReinforcementLearningModel:episodeUpdate(1)

		end
		
	end)
	
	NewGenerativeAdversarialImitationLearning:setDiagonalGaussianTrainFunction(function(previousFeatureMatrix, expertPreviousActionMeanMatrix, expertPreviousActionStandardDeviationMatrix, expertPreviousActionNoiseMatrix, currentFeatureMatrix, expertCurrentActionMeanMatrix, terminalStateMatrix)
		
		local DiscriminatorModel = NewGenerativeAdversarialImitationLearning.DiscriminatorModel

		local ReinforcementLearningModel = NewGenerativeAdversarialImitationLearning.ReinforcementLearningModel

		if (not DiscriminatorModel) then error("No discriminator neural network.") end

		if (not ReinforcementLearningModel) then error("No reinforcement learning neural network.") end

		local numberOfStepsPerEpisode = NewGenerativeAdversarialImitationLearning.numberOfStepsPerEpisode

		local isOutputPrinted = NewGenerativeAdversarialImitationLearning.isOutputPrinted

		local previousFeatureMatrixTable = NewGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(previousFeatureMatrix, numberOfStepsPerEpisode)

		local expertPreviousActionMeanMatrixTable = NewGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(expertPreviousActionMeanMatrix, numberOfStepsPerEpisode)

		local expertPreviousActionStandardDeviationTable = NewGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(expertPreviousActionStandardDeviationMatrix, numberOfStepsPerEpisode)
		
		local expertPreviousActionNoiseTable = NewGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(expertPreviousActionNoiseMatrix, numberOfStepsPerEpisode)

		local currentFeatureMatrixTable = NewGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(currentFeatureMatrix, numberOfStepsPerEpisode)
		
		local expertCurrentActionMeanMatrixTable = NewGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(expertCurrentActionMeanMatrix, numberOfStepsPerEpisode)
		
		local terminalStateMatrixTable = NewGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(terminalStateMatrix, numberOfStepsPerEpisode)

		local discriminatorInputNumberOfFeatures, discriminatorInputHasBias = DiscriminatorModel:getLayer(1)

		if (discriminatorInputNumberOfFeatures ~= (#expertPreviousActionMeanMatrix[1] + #previousFeatureMatrix[1])) then error("The number of input neurons for the discriminator does not match the total number of both state features and expert actions.") end

		discriminatorInputNumberOfFeatures = discriminatorInputNumberOfFeatures + ((discriminatorInputHasBias and 1) or 0)

		local discriminatorInputVector = AqwamTensorLibraryLinker:createTensor({1, discriminatorInputNumberOfFeatures}, 1)

		local currentEpisode = 1

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

				local concatenatedExpertStateActionVector = AqwamTensorLibraryLinker:concatenate(previousFeatureVector, expertPreviousActionMeanVector, 2)

				local concatenatedAgentStateActionVector = AqwamTensorLibraryLinker:concatenate(previousFeatureVector, agentPreviousActionMeanVector, 2)

				if (discriminatorInputHasBias) then

					table.insert(concatenatedExpertStateActionVector[1], 1)

					table.insert(concatenatedAgentStateActionVector[1], 1)

				end

				local discriminatorExpertActionValue = DiscriminatorModel:predict(concatenatedExpertStateActionVector, true)[1][1]

				local discriminatorAgentActionValue = DiscriminatorModel:predict(concatenatedAgentStateActionVector, true)[1][1]

				local discriminatorLoss = math.log(discriminatorExpertActionValue) + math.log(1 - discriminatorAgentActionValue)

				ReinforcementLearningModel:diagonalGaussianUpdate(previousFeatureVector, expertPreviousActionMeanVector, expertPreviousActionStandardDeviationVector, expertPreviousActionNoiseVector, discriminatorLoss, currentFeatureVector, expertCurrentActionMeanVector, terminalStateValue)

				DiscriminatorModel:forwardPropagate(discriminatorInputVector, true)

				DiscriminatorModel:update(discriminatorLoss, true)

				if (isOutputPrinted) then print("Episode: " .. currentEpisode .. "\t\tStep: " .. step .. "\t\tDiscriminator Loss: " .. discriminatorLoss) end

			end

			ReinforcementLearningModel:episodeUpdate(1)

		end
		
	end)
	
	return NewGenerativeAdversarialImitationLearning
	
end

return GenerativeAdversarialImitationLearning
