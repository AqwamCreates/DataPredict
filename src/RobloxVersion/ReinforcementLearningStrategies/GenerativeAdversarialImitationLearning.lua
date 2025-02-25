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

local AqwamTensorLibraryLinker = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local GenerativeAdversarialImitationLearningBaseModel = require(script.Parent.GenerativeAdversarialImitationLearningBaseModel)

GenerativeAdversarialImitationLearning = {}

GenerativeAdversarialImitationLearning.__index = GenerativeAdversarialImitationLearning

setmetatable(GenerativeAdversarialImitationLearning, GenerativeAdversarialImitationLearningBaseModel)

function GenerativeAdversarialImitationLearning.new(parameterDictionary)
	
	local NewGenerativeAdversarialImitationLearning = GenerativeAdversarialImitationLearningBaseModel.new(parameterDictionary)
	
	setmetatable(NewGenerativeAdversarialImitationLearning, GenerativeAdversarialImitationLearning)
	
	NewGenerativeAdversarialImitationLearning:setName("GenerativeAdversarialImitationLearning")
	
	NewGenerativeAdversarialImitationLearning:setCategoricalTrainFunction(function(previousFeatureMatrix, expertActionMatrix, currentFeatureMatrix, terminalStateMatrix)
		
		local DiscriminatorModel = NewGenerativeAdversarialImitationLearning.DiscriminatorModel

		local ReinforcementLearningModel = NewGenerativeAdversarialImitationLearning.ReinforcementLearningModel

		if (not DiscriminatorModel) then error("No discriminator neural network!") end

		if (not ReinforcementLearningModel) then error("No reinforcement learning neural network!") end

		local numberOfStepsPerEpisode = NewGenerativeAdversarialImitationLearning.numberOfStepsPerEpisode

		local isOutputPrinted = NewGenerativeAdversarialImitationLearning.isOutputPrinted

		local ClassesList = ReinforcementLearningModel:getClassesList()

		local previousFeatureMatrixTable = NewGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(previousFeatureMatrix, numberOfStepsPerEpisode)

		local expertActionMatrixTable = NewGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(expertActionMatrix, numberOfStepsPerEpisode)

		local currentFeatureMatrixTable = NewGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(currentFeatureMatrix, numberOfStepsPerEpisode)
		
		local terminalStateMatrixTable = NewGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(terminalStateMatrix, numberOfStepsPerEpisode)

		local discriminatorInputNumberOfFeatures, discriminatorInputHasBias = DiscriminatorModel:getLayer(1)

		if (discriminatorInputNumberOfFeatures ~= (#expertActionMatrix[1] + #previousFeatureMatrix[1])) then error("The number of input neurons for the discriminator does not match the total number of both state features and expert actions!") end

		discriminatorInputNumberOfFeatures = discriminatorInputNumberOfFeatures + ((discriminatorInputHasBias and 1) or 0)

		local discriminatorInputVector = AqwamTensorLibraryLinker:createTensor({1, discriminatorInputNumberOfFeatures}, 1)

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

				local concatenatedExpertStateActionVector = AqwamTensorLibraryLinker:concatenate(previousFeatureVector, expertActionVector, 2)

				local concatenatedAgentStateActionVector = AqwamTensorLibraryLinker:concatenate(previousFeatureVector, agentActionVector, 2)

				if (discriminatorInputHasBias) then

					table.insert(concatenatedExpertStateActionVector[1], 1)

					table.insert(concatenatedAgentStateActionVector[1], 1)

				end

				local discriminatorExpertActionValue = DiscriminatorModel:predict(concatenatedExpertStateActionVector, true)[1][1]

				local discriminatorAgentActionValue = DiscriminatorModel:predict(concatenatedAgentStateActionVector, true)[1][1]

				local discriminatorLoss = math.log(discriminatorExpertActionValue) + math.log(1 - discriminatorAgentActionValue)

				local actionIndex = NewGenerativeAdversarialImitationLearning:chooseIndexWithHighestValue(expertActionVector)

				local action = ClassesList[actionIndex]

				if (not action) then error("Missing action at index " .. actionIndex .. "!") end

				ReinforcementLearningModel:categoricalUpdate(previousFeatureVector, action, discriminatorLoss, currentFeatureVector, terminalStateValue)

				DiscriminatorModel:forwardPropagate(discriminatorInputVector, true)

				DiscriminatorModel:backwardPropagate(discriminatorLoss, true)

				if (isOutputPrinted) then print("Episode: " .. episode .. "\t\tStep: " .. step .. "\t\tDiscriminator Loss: " .. discriminatorLoss) end

			end

			ReinforcementLearningModel:episodeUpdate(1)

		end
		
	end)
	
	NewGenerativeAdversarialImitationLearning:setDiagonalGaussianTrainFunction(function(previousFeatureMatrix, expertActionMeanMatrix, expertActionStandardDeviationMatrix, expertActionNoiseMatrix, currentFeatureMatrix, terminalStateMatrix)
		
		local DiscriminatorModel = NewGenerativeAdversarialImitationLearning.DiscriminatorModel

		local ReinforcementLearningModel = NewGenerativeAdversarialImitationLearning.ReinforcementLearningModel

		if (not DiscriminatorModel) then error("No discriminator neural network!") end

		if (not ReinforcementLearningModel) then error("No reinforcement learning neural network!") end

		local numberOfStepsPerEpisode = NewGenerativeAdversarialImitationLearning.numberOfStepsPerEpisode

		local isOutputPrinted = NewGenerativeAdversarialImitationLearning.isOutputPrinted

		local previousFeatureMatrixTable = NewGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(previousFeatureMatrix, numberOfStepsPerEpisode)

		local expertActionMeanMatrixTable = NewGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(expertActionMeanMatrix, numberOfStepsPerEpisode)

		local expertActionStandardDeviationTable = NewGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(expertActionStandardDeviationMatrix, numberOfStepsPerEpisode)
		
		local expertActionNoiseTable = NewGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(expertActionNoiseMatrix, numberOfStepsPerEpisode)

		local currentFeatureMatrixTable = NewGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(currentFeatureMatrix, numberOfStepsPerEpisode)
		
		local terminalStateMatrixTable = NewGenerativeAdversarialImitationLearning:breakMatrixToMultipleSmallerMatrices(terminalStateMatrix, numberOfStepsPerEpisode)

		local discriminatorInputNumberOfFeatures, discriminatorInputHasBias = DiscriminatorModel:getLayer(1)

		if (discriminatorInputNumberOfFeatures ~= (#expertActionMeanMatrix[1] + #previousFeatureMatrix[1])) then error("The number of input neurons for the discriminator does not match the total number of both state features and expert actions!") end

		discriminatorInputNumberOfFeatures = discriminatorInputNumberOfFeatures + ((discriminatorInputHasBias and 1) or 0)

		local discriminatorInputVector = AqwamTensorLibraryLinker:createTensor({1, discriminatorInputNumberOfFeatures}, 1)

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

				local concatenatedExpertStateActionVector = AqwamTensorLibraryLinker:concatenate(previousFeatureVector, expertActionMeanVector, 2)

				local concatenatedAgentStateActionVector = AqwamTensorLibraryLinker:concatenate(previousFeatureVector, agentActionMeanVector, 2)

				if (discriminatorInputHasBias) then

					table.insert(concatenatedExpertStateActionVector[1], 1)

					table.insert(concatenatedAgentStateActionVector[1], 1)

				end

				local discriminatorExpertActionValue = DiscriminatorModel:predict(concatenatedExpertStateActionVector, true)[1][1]

				local discriminatorAgentActionValue = DiscriminatorModel:predict(concatenatedAgentStateActionVector, true)[1][1]

				local discriminatorLoss = math.log(discriminatorExpertActionValue) + math.log(1 - discriminatorAgentActionValue)

				ReinforcementLearningModel:diagonalGaussianUpdate(previousFeatureVector, expertActionMeanVector, expertActionStandardDeviationVector, expertActionNoiseVector, discriminatorLoss, currentFeatureVector, terminalStateValue)

				DiscriminatorModel:forwardPropagate(discriminatorInputVector, true)

				DiscriminatorModel:backwardPropagate(discriminatorLoss, true)

				if (isOutputPrinted) then print("Episode: " .. currentEpisode .. "\t\tStep: " .. step .. "\t\tDiscriminator Loss: " .. discriminatorLoss) end

			end

			ReinforcementLearningModel:episodeUpdate(1)

		end
		
	end)
	
	return NewGenerativeAdversarialImitationLearning
	
end

return GenerativeAdversarialImitationLearning