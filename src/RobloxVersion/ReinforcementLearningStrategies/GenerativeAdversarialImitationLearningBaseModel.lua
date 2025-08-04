--[[

	--------------------------------------------------------------------

	Aqwam's Machine, Deep And Reinforcement Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	Email: aqwam.harish.aiman@gmail.com
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict-Neural/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------
	
	DO NOT REMOVE THIS TEXT!
	
	--------------------------------------------------------------------

--]]

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local BaseModel = require(script.Parent.Parent.Models.BaseModel)

local GenerativeAdversarialImitationLearningBaseModel = {}

GenerativeAdversarialImitationLearningBaseModel.__index = GenerativeAdversarialImitationLearningBaseModel

setmetatable(GenerativeAdversarialImitationLearningBaseModel, BaseModel)

local defaultNumberOfStepsPerEpisode = 300

function GenerativeAdversarialImitationLearningBaseModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewGenerativeAdversarialImitationLearningBaseModel = BaseModel.new(parameterDictionary)
	
	setmetatable(NewGenerativeAdversarialImitationLearningBaseModel, GenerativeAdversarialImitationLearningBaseModel)
	
	NewGenerativeAdversarialImitationLearningBaseModel:setName("GenerativeAdversarialImitationLearningBaseModel")
	
	NewGenerativeAdversarialImitationLearningBaseModel:setClassName("GenerativeAdversarialImitationLearning")
	
	NewGenerativeAdversarialImitationLearningBaseModel.numberOfStepsPerEpisode = parameterDictionary.numberOfStepsPerEpisode or defaultNumberOfStepsPerEpisode
	
	NewGenerativeAdversarialImitationLearningBaseModel.categoricalTrainFunction = parameterDictionary.categoricalTrainFunction
	
	NewGenerativeAdversarialImitationLearningBaseModel.diagonalGaussianTrainFunction = parameterDictionary.diagonalGaussianTrainFunction
	
	NewGenerativeAdversarialImitationLearningBaseModel.DiscriminatorModel = parameterDictionary.DiscriminatorModel
	
	NewGenerativeAdversarialImitationLearningBaseModel.GeneratorModel = parameterDictionary.GeneratorModel
	
	return NewGenerativeAdversarialImitationLearningBaseModel
	
end

function GenerativeAdversarialImitationLearningBaseModel:setCategoricalTrainFunction(categoricalTrainFunction)
	
	self.categoricalTrainFunction = categoricalTrainFunction
	
end

function GenerativeAdversarialImitationLearningBaseModel:setDiagonalGaussianTrainFunction(diagonalGaussianTrainFunction)
	
	self.diagonalGaussianTrainFunction = diagonalGaussianTrainFunction
	
end

function GenerativeAdversarialImitationLearningBaseModel:categoricalTrain(previousFeatureMatrix, expertActionMatrix, currentFeatureMatrix)
	
	self.categoricalTrainFunction(previousFeatureMatrix, expertActionMatrix, currentFeatureMatrix)
	
end

function GenerativeAdversarialImitationLearningBaseModel:diagonalGaussianTrain(previousFeatureMatrix, expertActionMeanMatrix, expertActionStandardDeviationMatrix, expertActionNoiseMatrix, currentFeatureMatrix)
	
	self.diagonalGaussianTrainFunction(previousFeatureMatrix, expertActionMeanMatrix, expertActionStandardDeviationMatrix, expertActionNoiseMatrix, currentFeatureMatrix)
	
end

function GenerativeAdversarialImitationLearningBaseModel:evaluate(featureMatrix)

	return self.DiscriminatorModel:predict(featureMatrix, true)

end

function GenerativeAdversarialImitationLearningBaseModel:generate(featureMatrix, returnOriginalOutput)

	return self.ReinforcementLearningModel:predict(featureMatrix, returnOriginalOutput)

end

function GenerativeAdversarialImitationLearningBaseModel:setDiscriminatorModel(DiscriminatorModel)

	self.DiscriminatorModel = DiscriminatorModel

end

function GenerativeAdversarialImitationLearningBaseModel:setReinforcementLearningModel(ReinforcementLearningModel)

	self.ReinforcementLearningModel = ReinforcementLearningModel

end

function GenerativeAdversarialImitationLearningBaseModel:getDiscriminatorModel()

	return self.DiscriminatorModel

end

function GenerativeAdversarialImitationLearningBaseModel:getReinforcementLearningModel()

	return self.ReinforcementLearningModel

end

function GenerativeAdversarialImitationLearningBaseModel:chooseIndexWithHighestValue(valueVector)

	valueVector = valueVector[1]

	local highestValue = -math.huge

	local highestIndex

	for i, value in ipairs(valueVector) do

		if (value > highestValue) then

			highestValue = value

			highestIndex = i

		end

	end

	return highestIndex

end

function GenerativeAdversarialImitationLearningBaseModel:breakMatrixToMultipleSmallerMatrices(matrix, batchSize)

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

return GenerativeAdversarialImitationLearningBaseModel
