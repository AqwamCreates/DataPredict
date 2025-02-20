--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

ConditionalWassersteinGenerativeAdversarialNetworkModel = {}

ConditionalWassersteinGenerativeAdversarialNetworkModel.__index = ConditionalWassersteinGenerativeAdversarialNetworkModel

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

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

function ConditionalWassersteinGenerativeAdversarialNetworkModel.new(generatorMaximumNumberOfIterations, discriminatorMaximumNumberOfIterations, sampleSize)
	
	local NewConditionalWassersteinGenerativeAdversarialNetworkModel = {}
	
	setmetatable(NewConditionalWassersteinGenerativeAdversarialNetworkModel, ConditionalWassersteinGenerativeAdversarialNetworkModel)
	
	NewConditionalWassersteinGenerativeAdversarialNetworkModel.generatorMaximumNumberOfIterations = generatorMaximumNumberOfIterations or defaultGeneratorMaximumNumberOfIterations
	
	NewConditionalWassersteinGenerativeAdversarialNetworkModel.discriminatorMaximumNumberOfIterations = discriminatorMaximumNumberOfIterations or defaultDiscriminatorMaximumNumberOfIterations
	
	NewConditionalWassersteinGenerativeAdversarialNetworkModel.sampleSize = sampleSize or defaultSampleSize
	
	NewConditionalWassersteinGenerativeAdversarialNetworkModel.isOutputPrinted = true
	
	NewConditionalWassersteinGenerativeAdversarialNetworkModel.GeneratorModel = nil
	
	NewConditionalWassersteinGenerativeAdversarialNetworkModel.DiscriminatorModel = nil
	
	return NewConditionalWassersteinGenerativeAdversarialNetworkModel
	
end

function ConditionalWassersteinGenerativeAdversarialNetworkModel:setParameters(generatorMaximumNumberOfIterations, discriminatorMaximumNumberOfIterations, sampleSize)
	
	self.generatorMaximumNumberOfIterations = generatorMaximumNumberOfIterations or self.generatorMaximumNumberOfIterations

	self.discriminatorMaximumNumberOfIterations = discriminatorMaximumNumberOfIterations or self.discriminatorMaximumNumberOfIterations
	
	self.sampleSize = sampleSize or self.sampleSize
	
end

function ConditionalWassersteinGenerativeAdversarialNetworkModel:setDiscriminatorModel(DiscriminatorModel)
	
	self.DiscriminatorModel = DiscriminatorModel
	
end

function ConditionalWassersteinGenerativeAdversarialNetworkModel:setGeneratorModel(GeneratorModel)
	
	self.GeneratorModel = GeneratorModel
	
end

function ConditionalWassersteinGenerativeAdversarialNetworkModel:setPrintOutput(option)

	self.isOutputPrinted = option

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

	local discriminatorInputMatrix = AqwamMatrixLibrary:createMatrix(1, discriminatorInputNumberOfFeatures, 1)

	local generatorInputMatrix = AqwamMatrixLibrary:createMatrix(1, generatorInputNumberOfFeatures, 1)
	
	local functionToApplyToDiscriminator = function (discriminatorRealLabel, discriminatorGeneratedLabel) return -(discriminatorRealLabel - discriminatorGeneratedLabel) end
	
	local concatenatedNoiseFeatureMatrix = AqwamMatrixLibrary:horizontalConcatenate(noiseFeatureMatrix, labelMatrix)
	
	local concatenatedRealFeatureMatrix = AqwamMatrixLibrary:horizontalConcatenate(realFeatureMatrix, labelMatrix)
	
	local generatorNumberOfIterations = 0
	
	local discriminatorNumberOfIterations = 0
	
	local generatorMaximumNumberOfIterations = self.generatorMaximumNumberOfIterations
	
	local discriminatorMaximumNumberOfIterations = self.discriminatorMaximumNumberOfIterations
	
	local sampleSize = self.sampleSize
	
	local isOutputPrinted = self.isOutputPrinted

	repeat
		
		repeat

			task.wait()

			local realFeatureMatrixBatch, noiseFeatureMatrixBatch, labelMatrixBatch = sampleGroup(concatenatedRealFeatureMatrix, concatenatedNoiseFeatureMatrix, labelMatrix, sampleSize)

			local generatedLabelMatrixBatch = GeneratorModel:predict(noiseFeatureMatrixBatch, true)

			local concatenatedAndGeneratedLabelMatrix = AqwamMatrixLibrary:horizontalConcatenate(generatedLabelMatrixBatch, labelMatrixBatch)

			local discriminatorGeneratedLabelMatrix = DiscriminatorModel:predict(concatenatedAndGeneratedLabelMatrix, true)

			local discriminatorRealLabelMatrix = DiscriminatorModel:predict(realFeatureMatrixBatch, true)

			local meanDiscriminatorGeneratedLabelMatrix = AqwamMatrixLibrary:verticalMean(discriminatorGeneratedLabelMatrix)

			local meanDiscriminatorRealLabelMatrix = AqwamMatrixLibrary:verticalMean(discriminatorRealLabelMatrix)

			local discriminatorLossMatrix = AqwamMatrixLibrary:applyFunction(functionToApplyToDiscriminator, meanDiscriminatorRealLabelMatrix, meanDiscriminatorGeneratedLabelMatrix)

			DiscriminatorModel:forwardPropagate(discriminatorInputMatrix, true)

			DiscriminatorModel:backwardPropagate(discriminatorLossMatrix, true)

			discriminatorMaximumNumberOfIterations = discriminatorMaximumNumberOfIterations + 1

			if (isOutputPrinted) then print("Generator Iteration: " .. generatorNumberOfIterations .."\t\tDiscriminator Iteration: " .. discriminatorMaximumNumberOfIterations .. "\t\tDiscriminator Cost: " .. discriminatorLossMatrix[1][1]) end

		until (discriminatorNumberOfIterations >= discriminatorMaximumNumberOfIterations)

		local finalNoiseFeatureMatrixBatch, finalLabelMatrixBatch = samplePair(concatenatedNoiseFeatureMatrix, labelMatrix, sampleSize)

		local finalGeneratedLabelMatrix = GeneratorModel:predict(finalNoiseFeatureMatrixBatch, true)

		local finalConcatenatedAndGeneratedLabelMatrix = AqwamMatrixLibrary:horizontalConcatenate(finalGeneratedLabelMatrix, finalLabelMatrixBatch)

		local generatorLossMatrix = DiscriminatorModel:predict(finalConcatenatedAndGeneratedLabelMatrix, true)

		local meanGeneratorLossVector = AqwamMatrixLibrary:verticalMean(generatorLossMatrix)

		meanGeneratorLossVector = AqwamMatrixLibrary:multiply(-1, meanGeneratorLossVector)

		meanGeneratorLossVector = AqwamMatrixLibrary:createMatrix(1, generatorOutputNumberOfFeatures, meanGeneratorLossVector[1][1])

		GeneratorModel:forwardPropagate(generatorInputMatrix, true)

		GeneratorModel:backwardPropagate(meanGeneratorLossVector, true)
		
		generatorNumberOfIterations = generatorNumberOfIterations + 1
		
	until (generatorNumberOfIterations >= generatorMaximumNumberOfIterations)
	
end

function ConditionalWassersteinGenerativeAdversarialNetworkModel:evaluate(featureMatrix, labelMatrix)
	
	if (#featureMatrix ~= #labelMatrix) then error("The feature matrix and the label matrix must contain same number of data.") end
	
	local concatenatedMatrices = AqwamMatrixLibrary:horizontalConcatenate(featureMatrix, labelMatrix)
	
	return self.DiscriminatorModel:predict(concatenatedMatrices, true)
	
end

function ConditionalWassersteinGenerativeAdversarialNetworkModel:generate(noiseFeatureMatrix, labelMatrix)
	
	if (#noiseFeatureMatrix ~= #labelMatrix) then error("The noise feature matrix and the label matrix must contain same number of data.") end

	local concatenatedMatrices = AqwamMatrixLibrary:horizontalConcatenate(noiseFeatureMatrix, labelMatrix)
	
	return self.GeneratorModel:predict(concatenatedMatrices, true)
	
end

function ConditionalWassersteinGenerativeAdversarialNetworkModel:getDiscriminatorModel()
	
	return self.DiscriminatorModel
	
end

function ConditionalWassersteinGenerativeAdversarialNetworkModel:getGeneratorModel()

	return self.GeneratorModel

end

return ConditionalWassersteinGenerativeAdversarialNetworkModel