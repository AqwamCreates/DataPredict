ConditionalGenerativeAdversarialNetworkModel = {}

ConditionalGenerativeAdversarialNetworkModel.__index = ConditionalGenerativeAdversarialNetworkModel

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

local defaultMaxNumberOfIterations = 500

function ConditionalGenerativeAdversarialNetworkModel.new(maxNumberOfIterations)
	
	local NewConditionalGenerativeAdversarialNetworkModel = {}
	
	setmetatable(NewConditionalGenerativeAdversarialNetworkModel, ConditionalGenerativeAdversarialNetworkModel)
	
	NewConditionalGenerativeAdversarialNetworkModel.maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations
	
	NewConditionalGenerativeAdversarialNetworkModel.isOutputPrinted = true
	
	NewConditionalGenerativeAdversarialNetworkModel.Generator = nil
	
	NewConditionalGenerativeAdversarialNetworkModel.Discriminator = nil
	
	return NewConditionalGenerativeAdversarialNetworkModel
	
end

function ConditionalGenerativeAdversarialNetworkModel:setParameters(maxNumberOfIterations)
	
	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations
	
end

function ConditionalGenerativeAdversarialNetworkModel:setDiscriminator(Discriminator)
	
	self.Discriminator = Discriminator
	
end

function ConditionalGenerativeAdversarialNetworkModel:setGenerator(Generator)
	
	self.Generator = Generator
	
end

function ConditionalGenerativeAdversarialNetworkModel:setPrintOutput(option)
	
	if (option == false) then

		self.isOutputPrinted = false

	else

		self.isOutputPrinted = true

	end
	
end

function ConditionalGenerativeAdversarialNetworkModel:train(realFeatureMatrix, noiseFeatureMatrix, labelMatrix)
	
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
	
	local functionToApplyToDiscriminator = function (discriminatorRealLabel, discriminatorGeneratedLabel) return -(math.log(discriminatorRealLabel) + math.log(1 - discriminatorGeneratedLabel)) end
	
	local functionToApplyToGenerator = function (discriminatorGeneratedLabel) return math.log(1 - discriminatorGeneratedLabel) end
	
	local concatenatedNoiseFeatureMatrix = AqwamMatrixLibrary:horizontalConcatenate(noiseFeatureMatrix, labelMatrix)
	
	local concatenatedRealFeatureMatrix = AqwamMatrixLibrary:horizontalConcatenate(realFeatureMatrix, labelMatrix)
	
	local numberOfIterations = 0
	
	local maxNumberOfIterations = self.maxNumberOfIterations
	
	local isOutputPrinted = self.isOutputPrinted

	repeat
		
		task.wait()
		
		local generatedLabelMatrix = Generator:predict(concatenatedNoiseFeatureMatrix, true)
		
		local concatenatedAndGeneratedLabelMatrix = AqwamMatrixLibrary:horizontalConcatenate(generatedLabelMatrix, labelMatrix)
		
		local discriminatorGeneratedLabelMatrix = Discriminator:predict(concatenatedAndGeneratedLabelMatrix, true)
		
		local discriminatorRealLabelMatrix = Discriminator:predict(concatenatedRealFeatureMatrix, true)
		
		local discriminatorLossMatrix = AqwamMatrixLibrary:applyFunction(functionToApplyToDiscriminator, discriminatorRealLabelMatrix, discriminatorGeneratedLabelMatrix)
		
		local generatorLossMatrix = AqwamMatrixLibrary:applyFunction(functionToApplyToGenerator, discriminatorGeneratedLabelMatrix)
		
		local meanDiscriminatorLossMatrix = AqwamMatrixLibrary:verticalMean(discriminatorLossMatrix)
		
		local meanGeneratorLossVector = AqwamMatrixLibrary:verticalMean(generatorLossMatrix)
		
		meanGeneratorLossVector = AqwamMatrixLibrary:createMatrix(1, generatorOutputNumberOfFeatures, meanGeneratorLossVector[1][1])
		
		Discriminator:forwardPropagate(discriminatorInputMatrix, true)
		
		Discriminator:backPropagate(meanDiscriminatorLossMatrix, true)
		
		Generator:forwardPropagate(generatorInputMatrix, true)
		
		Generator:backPropagate(meanGeneratorLossVector, true)
		
		numberOfIterations = numberOfIterations + 1
		
		if (isOutputPrinted) then print("Iteration: " .. numberOfIterations .. "\t\tDiscriminator Cost: " .. meanDiscriminatorLossMatrix[1][1]) end
		
	until (numberOfIterations >= maxNumberOfIterations)
	
end

function ConditionalGenerativeAdversarialNetworkModel:evaluate(featureMatrix, labelMatrix)
	
	if (#featureMatrix ~= #labelMatrix) then error("The feature matrix and the label matrix must contain same number of data.") end
	
	local concatenatedMatrices = AqwamMatrixLibrary:horizontalConcatenate(featureMatrix, labelMatrix)
	
	return self.Discriminator:predict(concatenatedMatrices, true)
	
end

function ConditionalGenerativeAdversarialNetworkModel:generate(noiseFeatureMatrix, labelMatrix)
	
	if (#noiseFeatureMatrix ~= #labelMatrix) then error("The noise feature matrix and the label matrix must contain same number of data.") end

	local concatenatedMatrices = AqwamMatrixLibrary:horizontalConcatenate(noiseFeatureMatrix, labelMatrix)
	
	return self.Generator:predict(concatenatedMatrices, true)
	
end

function ConditionalGenerativeAdversarialNetworkModel:getGenerator()
	
	return self.Generator
	
end

function ConditionalGenerativeAdversarialNetworkModel:getDiscriminator()
	
	return self.Discriminator
	
end

return ConditionalGenerativeAdversarialNetworkModel
