ConditionalWassersteinGenerativeAdversarialNetworkModel = {}

ConditionalWassersteinGenerativeAdversarialNetworkModel.__index = ConditionalWassersteinGenerativeAdversarialNetworkModel

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local defaultMaxNumberOfIterations = 500

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

function ConditionalWassersteinGenerativeAdversarialNetworkModel.new(maxNumberOfIterations, sampleSize)
	
	local NewConditionalWassersteinGenerativeAdversarialNetworkModel = {}
	
	setmetatable(NewConditionalWassersteinGenerativeAdversarialNetworkModel, ConditionalWassersteinGenerativeAdversarialNetworkModel)
	
	NewConditionalWassersteinGenerativeAdversarialNetworkModel.maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations
	
	NewConditionalWassersteinGenerativeAdversarialNetworkModel.sampleSize = sampleSize or defaultSampleSize
	
	NewConditionalWassersteinGenerativeAdversarialNetworkModel.isOutputPrinted = true
	
	NewConditionalWassersteinGenerativeAdversarialNetworkModel.GeneratorModel = nil
	
	NewConditionalWassersteinGenerativeAdversarialNetworkModel.DiscriminatorModel = nil
	
	return NewConditionalWassersteinGenerativeAdversarialNetworkModel
	
end

function ConditionalWassersteinGenerativeAdversarialNetworkModel:setParameters(maxNumberOfIterations, sampleSize)
	
	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations
	
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
	
	local numberOfIterations = 0
	
	local maxNumberOfIterations = self.maxNumberOfIterations
	
	local isOutputPrinted = self.isOutputPrinted

	local sampleSize = self.sampleSize

	local isOutputPrinted = self.isOutputPrinted

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

		DiscriminatorModel:backPropagate(discriminatorLossMatrix, true)

		numberOfIterations = numberOfIterations + 1

		if (isOutputPrinted) then print("Iteration: " .. numberOfIterations .. "\t\tDiscriminator Cost: " .. discriminatorLossMatrix[1][1]) end

	until (numberOfIterations >= maxNumberOfIterations)

	local finalNoiseFeatureMatrixBatch, finalLabelMatrixBatch = samplePair(concatenatedNoiseFeatureMatrix, labelMatrix, sampleSize)

	local finalGeneratedLabelMatrix = GeneratorModel:predict(finalNoiseFeatureMatrixBatch, true)
	
	local finalConcatenatedAndGeneratedLabelMatrix = AqwamMatrixLibrary:horizontalConcatenate(finalGeneratedLabelMatrix, finalLabelMatrixBatch)
	
	local generatorLossMatrix = DiscriminatorModel:predict(finalConcatenatedAndGeneratedLabelMatrix, true)

	local meanGeneratorLossVector = AqwamMatrixLibrary:verticalMean(generatorLossMatrix)

	meanGeneratorLossVector = AqwamMatrixLibrary:multiply(-1, meanGeneratorLossVector)
	
	meanGeneratorLossVector = AqwamMatrixLibrary:createMatrix(1, generatorOutputNumberOfFeatures, meanGeneratorLossVector[1][1])

	GeneratorModel:forwardPropagate(generatorInputMatrix, true)

	GeneratorModel:backPropagate(meanGeneratorLossVector, true)
	
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
