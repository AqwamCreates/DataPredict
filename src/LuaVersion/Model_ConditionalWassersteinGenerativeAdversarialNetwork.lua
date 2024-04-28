ConditionalWassersteinGenerativeAdversarialNetworkModel = {}

ConditionalWassersteinGenerativeAdversarialNetworkModel.__index = ConditionalWassersteinGenerativeAdversarialNetworkModel

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

local defaultMaxNumberOfIterations = 500

local defaultSampleSize = 3

local function sample(matrix1, matrix2, sampleSize)

	local matrix1Batch = {}
	
	local matrix2Batch = {}

	local numberOfData = #matrix1

	for sample = 1, sampleSize, 1 do

		local randomIndex = Random.new():NextInteger(1, numberOfData)

		table.insert(matrix1Batch, matrix1[randomIndex])
		
		table.insert(matrix2Batch, matrix2[randomIndex])

	end

	return matrix1, matrix2

end

function ConditionalWassersteinGenerativeAdversarialNetworkModel.new(maxNumberOfIterations, sampleSize)
	
	local NewConditionalWassersteinGenerativeAdversarialNetworkModel = {}
	
	setmetatable(NewConditionalWassersteinGenerativeAdversarialNetworkModel, ConditionalWassersteinGenerativeAdversarialNetworkModel)
	
	NewConditionalWassersteinGenerativeAdversarialNetworkModel.maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations
	
	NewConditionalWassersteinGenerativeAdversarialNetworkModel.sampleSize = sampleSize or defaultSampleSize
	
	NewConditionalWassersteinGenerativeAdversarialNetworkModel.isOutputPrinted = true
	
	NewConditionalWassersteinGenerativeAdversarialNetworkModel.Generator = nil
	
	NewConditionalWassersteinGenerativeAdversarialNetworkModel.Discriminator = nil
	
	return NewConditionalWassersteinGenerativeAdversarialNetworkModel
	
end

function ConditionalWassersteinGenerativeAdversarialNetworkModel:setParameters(maxNumberOfIterations, sampleSize)
	
	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations
	
	self.sampleSize = sampleSize or self.sampleSize
	
end

function ConditionalWassersteinGenerativeAdversarialNetworkModel:setDiscriminator(Discriminator)
	
	self.Discriminator = Discriminator
	
end

function ConditionalWassersteinGenerativeAdversarialNetworkModel:setGenerator(Generator)
	
	self.Generator = Generator
	
end

function ConditionalWassersteinGenerativeAdversarialNetworkModel:setPrintOutput(option)
	
	if (option == false) then

		self.isOutputPrinted = false

	else

		self.isOutputPrinted = true

	end
	
end

function ConditionalWassersteinGenerativeAdversarialNetworkModel:train(realFeatureMatrix, noiseFeatureMatrix, labelMatrix)
	
	local Discriminator = self.Discriminator
	
	local Generator = self.Generator
	
	if (not Discriminator) then error("No discriminator neural network.") end
	
	if (not Generator) then error("No generator neural network.") end
	
	local numberOfFeaturesInLabelMatrix = #labelMatrix[1]
	
	local discriminatorNumberOfLayers = Generator:getNumberOfLayers()

	local generatorNumberOfLayers = Generator:getNumberOfLayers()
	
	local discriminatorInputNumberOfFeatures, discriminatorInputHasBias = Discriminator:getLayer(1)
	
	local generatorInputNumberOfFeatures, generatorInputHasBias = Generator:getLayer(1)
	
	local discriminatorOutputNumberOfFeatures, discriminatorOutputHasBias = Discriminator:getLayer(discriminatorNumberOfLayers)

	local generatorOutputNumberOfFeatures, generatorOutputHasBias = Generator:getLayer(generatorNumberOfLayers)
	
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

		local realFeatureMatrixBatch, noiseFeatureMatrixBatch = sample(concatenatedRealFeatureMatrix, concatenatedNoiseFeatureMatrix, sampleSize)

		local generatedLabelMatrixBatch = Generator:predict(noiseFeatureMatrix, true)

		local discriminatorGeneratedLabelMatrix = Discriminator:predict(generatedLabelMatrixBatch, true)

		local discriminatorRealLabelMatrix = Discriminator:predict(realFeatureMatrixBatch, true)

		local meanDiscriminatorGeneratedLabelMatrix = AqwamMatrixLibrary:verticalMean(discriminatorGeneratedLabelMatrix)

		local meanDiscriminatorRealLabelMatrix = AqwamMatrixLibrary:verticalMean(discriminatorRealLabelMatrix)

		local discriminatorLossMatrix = AqwamMatrixLibrary:applyFunction(functionToApplyToDiscriminator, meanDiscriminatorRealLabelMatrix, meanDiscriminatorGeneratedLabelMatrix)

		Discriminator:forwardPropagate(discriminatorInputMatrix, true)

		Discriminator:backPropagate(discriminatorLossMatrix, true)
		
		local discriminatorLossMatrix = AqwamMatrixLibrary:applyFunction(functionToApplyToDiscriminator, meanDiscriminatorRealLabelMatrix, meanDiscriminatorGeneratedLabelMatrix)

		local totalDiscriminatorCost = AqwamMatrixLibrary:power(discriminatorLossMatrix, 2)

		totalDiscriminatorCost = AqwamMatrixLibrary:sum(totalDiscriminatorCost)

		totalDiscriminatorCost = totalDiscriminatorCost / sampleSize

		numberOfIterations = numberOfIterations + 1

		if (isOutputPrinted) then print("Iteration: " .. numberOfIterations .. "\t\tDiscriminator Cost: " .. totalDiscriminatorCost) end

	until (numberOfIterations >= maxNumberOfIterations)

	local finalNoiseFeatureMatrixBatch = sample(noiseFeatureMatrix, sampleSize)

	local generatorLossMatrix = Generator:predict(finalNoiseFeatureMatrixBatch, true)

	local meanGeneratorLossVector = AqwamMatrixLibrary:verticalMean(generatorLossMatrix)

	meanGeneratorLossVector = AqwamMatrixLibrary:multiply(-1, meanGeneratorLossVector)

	Generator:forwardPropagate(generatorInputMatrix, true)

	Generator:backPropagate(meanGeneratorLossVector, true)
	
end

function ConditionalWassersteinGenerativeAdversarialNetworkModel:evaluate(featureMatrix, labelMatrix)
	
	if (#featureMatrix ~= #labelMatrix) then error("The feature matrix and the label matrix must contain same number of data.") end
	
	local concatenatedMatrices = AqwamMatrixLibrary:horizontalConcatenate(featureMatrix, labelMatrix)
	
	return self.Discriminator:predict(concatenatedMatrices, true)
	
end

function ConditionalWassersteinGenerativeAdversarialNetworkModel:generate(noiseFeatureMatrix, labelMatrix)
	
	if (#noiseFeatureMatrix ~= #labelMatrix) then error("The noise feature matrix and the label matrix must contain same number of data.") end

	local concatenatedMatrices = AqwamMatrixLibrary:horizontalConcatenate(noiseFeatureMatrix, labelMatrix)
	
	return self.Generator:predict(concatenatedMatrices, true)
	
end

function ConditionalWassersteinGenerativeAdversarialNetworkModel:getGenerator()
	
	return self.Generator
	
end

function ConditionalWassersteinGenerativeAdversarialNetworkModel:getDiscriminator()
	
	return self.Discriminator
	
end

return ConditionalWassersteinGenerativeAdversarialNetworkModel
