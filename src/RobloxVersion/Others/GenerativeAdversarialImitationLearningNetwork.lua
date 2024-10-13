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

GenerativeAdversarialNetworkModel = {}

GenerativeAdversarialNetworkModel.__index = GenerativeAdversarialNetworkModel

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local defaultNumberOfStepsPerEpisode = 300

local function chooseIndexWithHighestValue(vector)
	
	vector = vector[1]
	
	local highestValue = -math.huge
	
	local highestIndex
	
	for i, value in ipairs(vector) do
		
		if (value > highestValue) then
			
			highestValue = value
			
			highestIndex = i
			
		end
		
	end
	
	return highestIndex
	
end

local function breakMatrixToMultipleSmallerMatrices(matrix, batchSize)

	local numberOfBatches = math.ceil(#matrix/batchSize)

	local matrixBatchesTable = {}

	local batchPositions = {}

	local batchFeatureMatrix

	local batchLabelVector 

	for batch = 1, numberOfBatches, 1 do

		local startIndex = (batch - 1) * batchSize + 1

		local endIndex = math.min(batch * batchSize, #matrix)

		local batchFeatureMatrix = {}

		for i = startIndex, endIndex do table.insert(batchFeatureMatrix, matrix[i]) end

		table.insert(matrixBatchesTable, batchFeatureMatrix)

	end

	return matrixBatchesTable

end

function GenerativeAdversarialNetworkModel.new(numberOfStepsPerEpisode)
	
	local NewGenerativeAdversarialNetworkModel = {}
	
	setmetatable(NewGenerativeAdversarialNetworkModel, GenerativeAdversarialNetworkModel)
	
	NewGenerativeAdversarialNetworkModel.numberOfStepsPerEpisode = numberOfStepsPerEpisode or defaultNumberOfStepsPerEpisode
	
	NewGenerativeAdversarialNetworkModel.isOutputPrinted = true
	
	NewGenerativeAdversarialNetworkModel.ReinforcementLearningModel = nil
	
	NewGenerativeAdversarialNetworkModel.DiscriminatorModel = nil
	
	NewGenerativeAdversarialNetworkModel.ClassesList = {}
	
	return NewGenerativeAdversarialNetworkModel
	
end

function GenerativeAdversarialNetworkModel:setParameters(numberOfStepsPerEpisode)
	
	self.numberOfStepsPerEpisode = numberOfStepsPerEpisode or self.numberOfStepsPerEpisode
	
end

function GenerativeAdversarialNetworkModel:setDiscriminatorModel(DiscriminatorModel)
	
	self.DiscriminatorModel = DiscriminatorModel
	
end

function GenerativeAdversarialNetworkModel:setReinforcementLearningModel(ReinforcementLearningModel)
	
	self.ReinforcementLearningModel = ReinforcementLearningModel
	
end

function GenerativeAdversarialNetworkModel:setPrintOutput(option)

	self.isOutputPrinted = option

end

function GenerativeAdversarialNetworkModel:setClassesList(ClassesList)
	
	self.ClassesList = ClassesList
	
end

function GenerativeAdversarialNetworkModel:categoricalTrain(previousFeatureMatrix, expertActionMatrix, currentFeatureMatrix)
	
	local DiscriminatorModel = self.DiscriminatorModel
	
	local ReinforcementLearningModel = self.ReinforcementLearningModel
	
	if (not DiscriminatorModel) then error("No discriminator neural network!") end
	
	if (not ReinforcementLearningModel) then error("No reinforcement learning neural network!") end
	
	local numberOfStepsPerEpisode = self.numberOfStepsPerEpisode
	
	local isOutputPrinted = self.isOutputPrinted
	
	local ClassesList = self.ClassesList
	
	local previousFeatureMatrixTable = breakMatrixToMultipleSmallerMatrices(previousFeatureMatrix, numberOfStepsPerEpisode)
	
	local expertActionMatrixTable = breakMatrixToMultipleSmallerMatrices(expertActionMatrix, numberOfStepsPerEpisode)
	
	local currentFeatureMatrixTable = breakMatrixToMultipleSmallerMatrices(currentFeatureMatrix, numberOfStepsPerEpisode)
	
	local discriminatorInputNumberOfFeatures, discriminatorInputHasBias = DiscriminatorModel:getLayer(1)
	
	if (discriminatorInputNumberOfFeatures ~= (#expertActionMatrix[1] + #previousFeatureMatrix[1])) then error("The number of input neurons for the discriminator does not match the total number of both state features and expert actions.") end
	
	discriminatorInputNumberOfFeatures = discriminatorInputNumberOfFeatures + ((discriminatorInputHasBias and 1) or 0)
	
	local discriminatorInputVector = AqwamMatrixLibrary:createMatrix(1, discriminatorInputNumberOfFeatures, 1)
	
	for episode = 1, #previousFeatureMatrixTable, 1 do
		
		local previousFeatureSubMatrix = previousFeatureMatrixTable[episode]
		
		local expertActionSubMatrix = expertActionMatrixTable[episode]
		
		local currentFeatureSubMatrix = currentFeatureMatrixTable[episode]
		
		for step = 1, numberOfStepsPerEpisode, 1 do

			task.wait()
			
			local previousFeatureVector = {previousFeatureSubMatrix[step]}
			
			local expertActionVector = {expertActionSubMatrix[step]}
			
			local currentFeatureVector = {currentFeatureSubMatrix[step]}

			local agentActionVector = ReinforcementLearningModel:predict(previousFeatureVector, true)
			
			local concatenatedAgentStateActionVector = AqwamMatrixLibrary:horizontalConcatenate(previousFeatureVector, agentActionVector)
			
			local concatenatedExpertStateActionVector = AqwamMatrixLibrary:horizontalConcatenate(previousFeatureVector, expertActionVector)
			
			if (discriminatorInputHasBias) then
				
				table.insert(concatenatedAgentStateActionVector[1], 1)
				table.insert(concatenatedExpertStateActionVector[1], 1)
				
			end

			local discriminatorAgentActionValue = DiscriminatorModel:predict(concatenatedAgentStateActionVector, true)[1][1]

			local discriminatorExpertActionValue = DiscriminatorModel:predict(concatenatedExpertStateActionVector, true)[1][1]
			
			local discriminatorLoss = math.log(discriminatorExpertActionValue) + math.log(1 - discriminatorAgentActionValue)

			local actionIndex = chooseIndexWithHighestValue(expertActionVector)

			local action = ClassesList[actionIndex]
			
			if (not action) then error("Missing action at index " .. actionIndex .. ".") end

			ReinforcementLearningModel:categoricalUpdate(previousFeatureVector, action, discriminatorLoss, currentFeatureVector)

			DiscriminatorModel:forwardPropagate(discriminatorInputVector, true)

			DiscriminatorModel:backwardPropagate(-discriminatorLoss, true)

			if (isOutputPrinted) then print("Episode: " .. episode .. "\t\tStep: " .. step .. "\t\tDiscriminator Loss: " .. discriminatorLoss) end

		end
		
		ReinforcementLearningModel:episodeUpdate()
		
	end
	
end

function GenerativeAdversarialNetworkModel:diagonalGaussianTrain(previousFeatureMatrix, expertActionMeanMatrix, expertActionStandardDeviationMatrix, currentFeatureMatrix)

	local DiscriminatorModel = self.DiscriminatorModel

	local ReinforcementLearningModel = self.ReinforcementLearningModel

	if (not DiscriminatorModel) then error("No discriminator neural network!") end

	if (not ReinforcementLearningModel) then error("No reinforcement learning neural network!") end

	local numberOfStepsPerEpisode = self.numberOfStepsPerEpisode

	local isOutputPrinted = self.isOutputPrinted

	local ClassesList = self.ClassesList

	local previousFeatureMatrixTable = breakMatrixToMultipleSmallerMatrices(previousFeatureMatrix, numberOfStepsPerEpisode)

	local expertActionMeanMatrixTable = breakMatrixToMultipleSmallerMatrices(expertActionMeanMatrix, numberOfStepsPerEpisode)
	
	local expertActionStandardDeviationTable = breakMatrixToMultipleSmallerMatrices(expertActionStandardDeviationMatrix, numberOfStepsPerEpisode)

	local currentFeatureMatrixTable = breakMatrixToMultipleSmallerMatrices(currentFeatureMatrix, numberOfStepsPerEpisode)
	
	local discriminatorInputNumberOfFeatures, discriminatorInputHasBias = DiscriminatorModel:getLayer(1)
	
	if (discriminatorInputNumberOfFeatures ~= (#expertActionMeanMatrix[1] + #previousFeatureMatrix[1])) then error("The number of input neurons for the discriminator does not match the total number of both state features and expert actions.") end

	discriminatorInputNumberOfFeatures = discriminatorInputNumberOfFeatures + ((discriminatorInputHasBias and 1) or 0)

	local discriminatorInputVector = AqwamMatrixLibrary:createMatrix(1, discriminatorInputNumberOfFeatures, 1)

	local currentEpisode = 1

	for episode = 1, #previousFeatureMatrixTable, 1 do

		local previousFeatureSubMatrix = previousFeatureMatrixTable[episode]

		local expertActionMeanSubMatrix = expertActionMeanMatrixTable[episode]
		
		local expertActionStandardDeviationSubMatrix = expertActionStandardDeviationTable[episode]

		local currentFeatureSubMatrix = currentFeatureMatrixTable[episode]

		for step = 1, numberOfStepsPerEpisode, 1 do

			task.wait()

			local previousFeatureVector = {previousFeatureSubMatrix[step]}

			local expertActionMeanVector = {expertActionMeanSubMatrix[step]}
			
			local expertActionStandardDeviationVector = {expertActionStandardDeviationSubMatrix[step]}

			local currentFeatureVector = {currentFeatureSubMatrix[step]}

			local agentActionMeanVector = ReinforcementLearningModel:predict(previousFeatureVector, true)
			
			local concatenatedAgentStateActionVector = AqwamMatrixLibrary:horizontalConcatenate(previousFeatureVector, agentActionMeanVector)

			local concatenatedExpertStateActionVector = AqwamMatrixLibrary:horizontalConcatenate(previousFeatureVector, expertActionMeanVector)

			if (discriminatorInputHasBias) then

				table.insert(concatenatedAgentStateActionVector[1], 1)
				table.insert(concatenatedExpertStateActionVector[1], 1)

			end

			local discriminatorAgentActionValue = DiscriminatorModel:predict(concatenatedAgentStateActionVector, true)[1][1]

			local discriminatorExpertActionValue = DiscriminatorModel:predict(concatenatedExpertStateActionVector, true)[1][1]

			local discriminatorLoss = math.log(discriminatorExpertActionValue) + math.log(1 - discriminatorAgentActionValue)

			ReinforcementLearningModel:diagonalGaussianUpdate(previousFeatureVector, expertActionMeanVector, expertActionStandardDeviationVector, discriminatorLoss, currentFeatureVector)

			DiscriminatorModel:forwardPropagate(discriminatorInputVector, true)

			DiscriminatorModel:backwardPropagate(-discriminatorLoss, true)

			if (isOutputPrinted) then print("Episode: " .. currentEpisode .. "\t\tStep: " .. step .. "\t\tDiscriminator Loss: " .. discriminatorLoss) end

		end
		
		ReinforcementLearningModel:episodeUpdate()

	end

end

function GenerativeAdversarialNetworkModel:evaluate(featureMatrix)
	
	return self.DiscriminatorModel:predict(featureMatrix, true)
	
end

function GenerativeAdversarialNetworkModel:generate(noiseFeatureMatrix, returnOriginalOutput)
	
	return self.ReinforcementLearningModel:predict(noiseFeatureMatrix, returnOriginalOutput)
	
end

function GenerativeAdversarialNetworkModel:getDiscriminatorModel()

	return self.DiscriminatorModel

end

function GenerativeAdversarialNetworkModel:getReinforcementLearningModel()

	return self.ReinforcementLearningModel

end

function GenerativeAdversarialNetworkModel:getClassesList()

	return self.ClassesList

end

return GenerativeAdversarialNetworkModel
