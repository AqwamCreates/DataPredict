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

local AqwamTensorLibrary = require("AqwamTensorLibrary")

local GenerativeAdversarialNetworkBaseModel = require("Model_GenerativeAdversarialNetworkBaseModel")

ConditionalWassersteinGenerativeAdversarialNetworkModel = {}

ConditionalWassersteinGenerativeAdversarialNetworkModel.__index = ConditionalWassersteinGenerativeAdversarialNetworkModel

setmetatable(ConditionalWassersteinGenerativeAdversarialNetworkModel, GenerativeAdversarialNetworkBaseModel)

local defaultGeneratorMaximumNumberOfIterations = 50

local defaultDiscriminatorMaximumNumberOfIterations = 100

local defaultSampleSize = 3

local function sampleGroup(matrix1, matrix2, matrix3, sampleSize)

	local matrix1Batch = {}
	
	local matrix2Batch = {}
	
	local matrix3Batch = {}

	local numberOfData = #matrix1

	for sample = 1, sampleSize, 1 do

		local randomIndex = Random.new():NextInteger(1, numberOfData)

		table.insert(matrix1Batch, matrix1[randomIndex])
		
		table.insert(matrix2Batch, matrix2[randomIndex])
		
		table.insert(matrix3Batch, matrix3[randomIndex])

	end

	return matrix1Batch, matrix2Batch, matrix3Batch

end

local function samplePair(matrix, matrix2, sampleSize)

	local matrix1Batch = {}
	
	local matrix2Batch = {}

	local numberOfData = #matrix

	for sample = 1, sampleSize, 1 do

		local randomIndex = Random.new():NextInteger(1, numberOfData)

		table.insert(matrix1Batch, matrix[randomIndex])
		
		table.insert(matrix2Batch, matrix2[randomIndex])


	end

	return matrix1Batch, matrix2Batch

end

function ConditionalWassersteinGenerativeAdversarialNetworkModel.new(parameterDictionary)
	
	local NewConditionalWassersteinGenerativeAdversarialNetworkModel = GenerativeAdversarialNetworkBaseModel.new(parameterDictionary)
	
	setmetatable(NewConditionalWassersteinGenerativeAdversarialNetworkModel, ConditionalWassersteinGenerativeAdversarialNetworkModel)
	
	NewConditionalWassersteinGenerativeAdversarialNetworkModel:setName("ConditionalWassersteinGenerativeAdversarialNetwork")
	
	NewConditionalWassersteinGenerativeAdversarialNetworkModel.generatorMaximumNumberOfIterations = parameterDictionary.generatorMaximumNumberOfIterations or defaultGeneratorMaximumNumberOfIterations
	
	NewConditionalWassersteinGenerativeAdversarialNetworkModel.discriminatorMaximumNumberOfIterations = parameterDictionary.discriminatorMaximumNumberOfIterations or defaultDiscriminatorMaximumNumberOfIterations
	
	NewConditionalWassersteinGenerativeAdversarialNetworkModel.sampleSize = parameterDictionary.sampleSize or defaultSampleSize
	
	return NewConditionalWassersteinGenerativeAdversarialNetworkModel
	
end

function ConditionalWassersteinGenerativeAdversarialNetworkModel:train(realFeatureMatrix, noiseFeatureMatrix, labelMatrix)
	
	local DiscriminatorModel = self.DiscriminatorModel
	
	local GeneratorModel = self.GeneratorModel
	
	if (not DiscriminatorModel) then error("No discriminator neural network.") end
	
	if (not GeneratorModel) then error("No generator neural network.") end
	
	local numberOfFeaturesInLabelMatrix = #labelMatrix[1]
	
	local discriminatorNumberOfLayers = GeneratorModel:getNumberOfLayers()

	local generatorNumberOfLayers = GeneratorModel:getNumberOfLayers()
	
	local discriminatorInputNumberOfFeatures, discriminatorInputHasBias = DiscriminatorModel:getLayer(1)
	
	local generatorInputNumberOfFeatures, generatorInputHasBias = GeneratorModel:getLayer(1)
	
	local discriminatorOutputNumberOfFeatures, discriminatorOutputHasBias = DiscriminatorModel:getLayer(discriminatorNumberOfLayers)

	local generatorOutputNumberOfFeatures, generatorOutputHasBias = GeneratorModel:getLayer(generatorNumberOfLayers)
	
	discriminatorInputNumberOfFeatures = discriminatorInputNumberOfFeatures + ((discriminatorInputHasBias and 1) or 0)

	generatorInputNumberOfFeatures = generatorInputNumberOfFeatures + ((generatorInputHasBias and 1) or 0)
	
	discriminatorOutputNumberOfFeatures = discriminatorOutputNumberOfFeatures + ((discriminatorOutputHasBias and 1) or 0)
	
	generatorOutputNumberOfFeatures = generatorOutputNumberOfFeatures + ((generatorOutputHasBias and 1) or 0)
	
	if ((generatorOutputNumberOfFeatures + numberOfFeaturesInLabelMatrix) ~= discriminatorInputNumberOfFeatures) then error("The number of neurons at the discriminator's input layer must equal to the total of number of neurons at the generator's output layer and the number of features in label matrix.") end
	
	if (discriminatorOutputNumberOfFeatures ~= 1) then error("The number of neurons at the discriminator's output layer must be equal to 1.") end
	
	if (#realFeatureMatrix ~= #noiseFeatureMatrix) or (#realFeatureMatrix ~= #labelMatrix) then error("All matrices must contain same number of data.") end
	
	if ((#noiseFeatureMatrix[1] + numberOfFeaturesInLabelMatrix)  ~= generatorInputNumberOfFeatures) then error("The total number of columns in noise feature matrix and label matrix must contain the same number as the number of neurons in generator's input layer.") end
	
	if ((#realFeatureMatrix[1] + numberOfFeaturesInLabelMatrix) ~= discriminatorInputNumberOfFeatures) then error("The total number of columns in real feature matrix and label matrix must contain the same number as the number of neurons in discriminator's input layer.") end

	local discriminatorInputMatrix = AqwamTensorLibrary:createTensor({1, discriminatorInputNumberOfFeatures}, 1)

	local generatorInputMatrix = AqwamTensorLibrary:createTensor({1, generatorInputNumberOfFeatures}, 1)
	
	local functionToApplyToDiscriminator = function (discriminatorRealLabel, discriminatorGeneratedLabel) return -(discriminatorRealLabel - discriminatorGeneratedLabel) end
	
	local concatenatedNoiseFeatureMatrix = AqwamTensorLibrary:concatenate(noiseFeatureMatrix, labelMatrix, 2)
	
	local concatenatedRealFeatureMatrix = AqwamTensorLibrary:concatenate(realFeatureMatrix, labelMatrix,2)
	
	local generatorNumberOfIterations = 0
	
	local discriminatorNumberOfIterations = 0
	
	local generatorMaximumNumberOfIterations = self.generatorMaximumNumberOfIterations
	
	local discriminatorMaximumNumberOfIterations = self.discriminatorMaximumNumberOfIterations
	
	local sampleSize = self.sampleSize
	
	local isOutputPrinted = self.isOutputPrinted
	
	local meanDiscriminatorLossValue = 0

	repeat
		
		repeat

			task.wait()

			local realFeatureMatrixBatch, noiseFeatureMatrixBatch, labelMatrixBatch = sampleGroup(concatenatedRealFeatureMatrix, concatenatedNoiseFeatureMatrix, labelMatrix, sampleSize)

			local generatedLabelMatrixBatch = GeneratorModel:predict(noiseFeatureMatrixBatch, true)

			local concatenatedAndGeneratedLabelMatrix = AqwamTensorLibrary:concatenate(generatedLabelMatrixBatch, labelMatrixBatch, 2)

			local discriminatorGeneratedLabelMatrix = DiscriminatorModel:predict(concatenatedAndGeneratedLabelMatrix, true)

			local discriminatorRealLabelMatrix = DiscriminatorModel:predict(realFeatureMatrixBatch, true)

			local meanDiscriminatorGeneratedLabelMatrix = AqwamTensorLibrary:mean(discriminatorGeneratedLabelMatrix, 1)

			local meanDiscriminatorRealLabelMatrix = AqwamTensorLibrary:mean(discriminatorRealLabelMatrix, 1)

			local discriminatorLossMatrix = AqwamTensorLibrary:applyFunction(functionToApplyToDiscriminator, meanDiscriminatorRealLabelMatrix, meanDiscriminatorGeneratedLabelMatrix)

			DiscriminatorModel:forwardPropagate(discriminatorInputMatrix, true)

			DiscriminatorModel:update(discriminatorLossMatrix, true)

			discriminatorMaximumNumberOfIterations = discriminatorMaximumNumberOfIterations + 1
			
			meanDiscriminatorLossValue = discriminatorLossMatrix[1][1]

			if (isOutputPrinted) then print("Generator Iteration: " .. generatorNumberOfIterations .."\t\tDiscriminator Iteration: " .. discriminatorMaximumNumberOfIterations .. "\t\tDiscriminator Cost: " .. meanDiscriminatorLossValue) end

		until (discriminatorNumberOfIterations >= discriminatorMaximumNumberOfIterations) or self:checkIfTargetCostReached(meanDiscriminatorLossValue) or self:checkIfConverged(meanDiscriminatorLossValue) 

		local finalNoiseFeatureMatrixBatch, finalLabelMatrixBatch = samplePair(concatenatedNoiseFeatureMatrix, labelMatrix, sampleSize)

		local finalGeneratedLabelMatrix = GeneratorModel:predict(finalNoiseFeatureMatrixBatch, true)

		local finalConcatenatedAndGeneratedLabelMatrix = AqwamTensorLibrary:concatenate(finalGeneratedLabelMatrix, finalLabelMatrixBatch)

		local generatorLossMatrix = DiscriminatorModel:predict(finalConcatenatedAndGeneratedLabelMatrix, true)

		local meanGeneratorLossVector = AqwamTensorLibrary:mean(generatorLossMatrix, 1)

		meanGeneratorLossVector = AqwamTensorLibrary:multiply(-1, meanGeneratorLossVector)

		meanGeneratorLossVector = AqwamTensorLibrary:createTensor({1, generatorOutputNumberOfFeatures}, meanGeneratorLossVector[1][1])

		GeneratorModel:forwardPropagate(generatorInputMatrix, true)

		GeneratorModel:update(meanGeneratorLossVector, true)
		
		generatorNumberOfIterations = generatorNumberOfIterations + 1
		
	until (generatorNumberOfIterations >= generatorMaximumNumberOfIterations)
	
end

function ConditionalWassersteinGenerativeAdversarialNetworkModel:evaluate(featureMatrix, labelMatrix)

	local DiscriminatorModel = self.DiscriminatorModel

	if (not DiscriminatorModel) then error("No discriminator neural network.") end

	if (#featureMatrix ~= #labelMatrix) then error("The feature matrix and the label matrix must contain same number of data.") end

	local concatenatedMatrices = AqwamTensorLibrary:concatenate(featureMatrix, labelMatrix, 2)

	return DiscriminatorModel:predict(concatenatedMatrices, true)

end

function ConditionalWassersteinGenerativeAdversarialNetworkModel:generate(noiseFeatureMatrix, labelMatrix)

	local GeneratorModel =  self.GeneratorModel

	if (not GeneratorModel) then error("No generator neural network.") end

	if (#noiseFeatureMatrix ~= #labelMatrix) then error("The noise feature matrix and the label matrix must contain same number of data.") end

	local concatenatedMatrices = AqwamTensorLibrary:concatenate(noiseFeatureMatrix, labelMatrix, 2)

	return GeneratorModel:predict(concatenatedMatrices, true)

end

return ConditionalWassersteinGenerativeAdversarialNetworkModel