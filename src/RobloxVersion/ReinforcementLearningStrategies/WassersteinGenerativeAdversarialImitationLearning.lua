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

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local GenerativeAdversarialImitationLearningBaseModel = require(script.Parent.GenerativeAdversarialImitationLearningBaseModel)

WassersteinGenerativeAdversarialImitationLearning = {}

WassersteinGenerativeAdversarialImitationLearning.__index = WassersteinGenerativeAdversarialImitationLearning

setmetatable(WassersteinGenerativeAdversarialImitationLearning, GenerativeAdversarialImitationLearningBaseModel)

function WassersteinGenerativeAdversarialImitationLearning.new(parameterDictionary)
	
	local NewWassersteinGenerativeAdversarialImitationLearning = GenerativeAdversarialImitationLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewWassersteinGenerativeAdversarialImitationLearning, WassersteinGenerativeAdversarialImitationLearning)
	
	NewWassersteinGenerativeAdversarialImitationLearning:setName("WassersteinGenerativeAdversarialImitationLearning")
	
	NewWassersteinGenerativeAdversarialImitationLearning:setCategoricalTrainFunction(function(previousFeatureMatrix, expertActionMatrix, currentFeatureMatrix, terminalStateMatrix)
		
		local DiscriminatorModel = NewWassersteinGenerativeAdversarialImitationLearning.DiscriminatorModel

		local ReinforcementLearningModel = NewWassersteinGenerativeAdversarialImitationLearning.ReinforcementLearningModel

		if (not DiscriminatorModel) then error("No discriminator neural network!") end

		if (not ReinforcementLearningModel) then error("No reinforcement learning neural network!") end

		local numberOfStepsPerEpisode = NewWassersteinGenerativeAdversarialImitationLearning.numberOfStepsPerEpisode

		local isOutputPrinted = NewWassersteinGenerativeAdversarialImitationLearning.isOutputPrinted

		local ActionsList = ReinforcementLearningModel:getActionsList()

		local previousFeatureMatrixTable = NewWassersteinGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(previousFeatureMatrix, numberOfStepsPerEpisode)

		local expertActionMatrixTable = NewWassersteinGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(expertActionMatrix, numberOfStepsPerEpisode)

		local currentFeatureMatrixTable = NewWassersteinGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(currentFeatureMatrix, numberOfStepsPerEpisode)
		
		local terminalStateMatrixTable = NewWassersteinGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(terminalStateMatrix, numberOfStepsPerEpisode)

		local discriminatorInputNumberOfFeatures, discriminatorInputHasBias = DiscriminatorModel:getLayer(1)

		if (discriminatorInputNumberOfFeatures ~= (#expertActionMatrix[1] + #previousFeatureMatrix[1])) then error("The number of input neurons for the discriminator does not match the total number of both state features and expert actions!") end

		discriminatorInputNumberOfFeatures = discriminatorInputNumberOfFeatures + ((discriminatorInputHasBias and 1) or 0)

		local discriminatorInputVector = AqwamTensorLibrary:createTensor({1, discriminatorInputNumberOfFeatures}, 1)

		for episode = 1, #previousFeatureMatrixTable, 1 do

			local previousFeatureSubMatrix = previousFeatureMatrixTable[episode]

			local expertActionSubMatrix = expertActionMatrixTable[episode]

			local currentFeatureSubMatrix = currentFeatureMatrixTable[episode]
			
			local terminalStateSubMatrix = terminalStateMatrixTable[episode]

			for step = 1, numberOfStepsPerEpisode, 1 do

				task.wait()

				local previousFeatureVector = {previousFeatureSubMatrix[step]}

				local expertActionVector = {expertActionSubMatrix[step]}

				local currentFeatureVector = {currentFeatureSubMatrix[step]}
				
				local terminalStateValue = terminalStateSubMatrix[step][1]

				local agentActionVector = ReinforcementLearningModel:predict(previousFeatureVector, true)

				local concatenatedExpertStateActionVector = AqwamTensorLibrary:concatenate(previousFeatureVector, expertActionVector, 2)

				local concatenatedAgentStateActionVector = AqwamTensorLibrary:concatenate(previousFeatureVector, agentActionVector, 2)

				if (discriminatorInputHasBias) then

					table.insert(concatenatedExpertStateActionVector[1], 1)

					table.insert(concatenatedAgentStateActionVector[1], 1)

				end

				local discriminatorExpertActionValue = DiscriminatorModel:predict(concatenatedExpertStateActionVector, true)[1][1]

				local discriminatorAgentActionValue = DiscriminatorModel:predict(concatenatedAgentStateActionVector, true)[1][1]

				local discriminatorLoss = discriminatorExpertActionValue - discriminatorAgentActionValue

				local actionIndex = NewWassersteinGenerativeAdversarialImitationLearning:chooseIndexWithHighestValue(expertActionVector)

				local action = ActionsList[actionIndex]

				if (not action) then error("Missing action at index " .. actionIndex .. "!") end

				ReinforcementLearningModel:categoricalUpdate(previousFeatureVector, action, discriminatorLoss, currentFeatureVector, terminalStateValue)

				DiscriminatorModel:forwardPropagate(discriminatorInputVector, true)

				DiscriminatorModel:backwardPropagate(discriminatorLoss, true)

				if (isOutputPrinted) then print("Episode: " .. episode .. "\t\tStep: " .. step .. "\t\tDiscriminator Loss: " .. discriminatorLoss) end

			end

			ReinforcementLearningModel:episodeUpdate(1)

		end
		
		
	end)
	
	NewWassersteinGenerativeAdversarialImitationLearning:setDiagonalGaussianTrainFunction(function(previousFeatureMatrix, expertActionMeanMatrix, expertActionStandardDeviationMatrix, expertActionNoiseMatrix, currentFeatureMatrix, terminalStateMatrix)
		
		local DiscriminatorModel = NewWassersteinGenerativeAdversarialImitationLearning.DiscriminatorModel

		local ReinforcementLearningModel = NewWassersteinGenerativeAdversarialImitationLearning.ReinforcementLearningModel

		if (not DiscriminatorModel) then error("No discriminator neural network!") end

		if (not ReinforcementLearningModel) then error("No reinforcement learning neural network!") end

		local numberOfStepsPerEpisode = NewWassersteinGenerativeAdversarialImitationLearning.numberOfStepsPerEpisode

		local isOutputPrinted = NewWassersteinGenerativeAdversarialImitationLearning.isOutputPrinted

		local previousFeatureMatrixTable = NewWassersteinGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(previousFeatureMatrix, numberOfStepsPerEpisode)

		local expertActionMeanMatrixTable = NewWassersteinGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(expertActionMeanMatrix, numberOfStepsPerEpisode)

		local expertActionStandardDeviationTable = NewWassersteinGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(expertActionStandardDeviationMatrix, numberOfStepsPerEpisode)
		
		local expertActionNoiseTable = NewWassersteinGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(expertActionNoiseMatrix, numberOfStepsPerEpisode)

		local currentFeatureMatrixTable = NewWassersteinGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(currentFeatureMatrix, numberOfStepsPerEpisode)
		
		local terminalStateMatrixTable = NewWassersteinGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(terminalStateMatrix, numberOfStepsPerEpisode)

		local discriminatorInputNumberOfFeatures, discriminatorInputHasBias = DiscriminatorModel:getLayer(1)

		if (discriminatorInputNumberOfFeatures ~= (#expertActionMeanMatrix[1] + #previousFeatureMatrix[1])) then error("The number of input neurons for the discriminator does not match the total number of both state features and expert actions!") end

		discriminatorInputNumberOfFeatures = discriminatorInputNumberOfFeatures + ((discriminatorInputHasBias and 1) or 0)

		local discriminatorInputVector = AqwamTensorLibrary:createTensor({1, discriminatorInputNumberOfFeatures}, 1)

		local currentEpisode = 1

		for episode = 1, #previousFeatureMatrixTable, 1 do

			local previousFeatureSubMatrix = previousFeatureMatrixTable[episode]

			local expertActionMeanSubMatrix = expertActionMeanMatrixTable[episode]

			local expertActionStandardDeviationSubMatrix = expertActionStandardDeviationTable[episode]
			
			local expertActionNoiseSubMatrix = expertActionNoiseTable[episode]

			local currentFeatureSubMatrix = currentFeatureMatrixTable[episode]
			
			local terminalStateSubMatrix = terminalStateMatrixTable[episode]

			for step = 1, numberOfStepsPerEpisode, 1 do

				task.wait()

				local previousFeatureVector = {previousFeatureSubMatrix[step]}

				local expertActionMeanVector = {expertActionMeanSubMatrix[step]}

				local expertActionStandardDeviationVector = {expertActionStandardDeviationSubMatrix[step]}
				
				local expertActionNoiseVector = {expertActionNoiseSubMatrix[step]}

				local currentFeatureVector = {currentFeatureSubMatrix[step]}
				
				local terminalStateValue = terminalStateSubMatrix[step][1]

				local agentActionMeanVector = ReinforcementLearningModel:predict(previousFeatureVector, true)

				local concatenatedExpertStateActionVector = AqwamTensorLibrary:concatenate(previousFeatureVector, expertActionMeanVector, 2)

				local concatenatedAgentStateActionVector = AqwamTensorLibrary:concatenate(previousFeatureVector, agentActionMeanVector, 2)

				if (discriminatorInputHasBias) then

					table.insert(concatenatedExpertStateActionVector[1], 1)

					table.insert(concatenatedAgentStateActionVector[1], 1)

				end

				local discriminatorExpertActionValue = DiscriminatorModel:predict(concatenatedExpertStateActionVector, true)[1][1]

				local discriminatorAgentActionValue = DiscriminatorModel:predict(concatenatedAgentStateActionVector, true)[1][1]

				local discriminatorLoss = discriminatorExpertActionValue - discriminatorAgentActionValue

				ReinforcementLearningModel:diagonalGaussianUpdate(previousFeatureVector, expertActionMeanVector, expertActionStandardDeviationVector, expertActionNoiseVector, discriminatorLoss, currentFeatureVector, terminalStateValue)

				DiscriminatorModel:forwardPropagate(discriminatorInputVector, true)

				DiscriminatorModel:update(discriminatorLoss, true)

				if (isOutputPrinted) then print("Episode: " .. currentEpisode .. "\t\tStep: " .. step .. "\t\tDiscriminator Loss: " .. discriminatorLoss) end

			end

			ReinforcementLearningModel:episodeUpdate(1)

		end
		
	end)
	
	return NewWassersteinGenerativeAdversarialImitationLearning
	
end

return WassersteinGenerativeAdversarialImitationLearning
