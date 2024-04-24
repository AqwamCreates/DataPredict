WassersteinGenerativeAdversarialNetworkModel = {}

WassersteinGenerativeAdversarialNetworkModel.__index = WassersteinGenerativeAdversarialNetworkModel

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local defaultMaxNumberOfIterations = 500

local defaultSampleSize = 3

local function samplePair(realFeatureMatrix, noiseFeatureMatrix, sampleSize)
	
	local realFeatureMatrixBatch = {}
	
	local noiseFeatureMatrixBatch = {}
	
	local numberOfData = #realFeatureMatrixBatch
	
	for sample = 1, sampleSize, 1 do
		
		local randomIndex = Random.new():NextInteger(1, numberOfData)
		
		table.insert(realFeatureMatrixBatch, realFeatureMatrix[randomIndex])
		table.insert(noiseFeatureMatrixBatch, noiseFeatureMatrix[randomIndex])
		
	end
	
	return realFeatureMatrixBatch, noiseFeatureMatrixBatch
	
end

local function sample(noiseFeatureMatrix, sampleSize)

	local noiseFeatureMatrixBatch = {}

	local numberOfData = #noiseFeatureMatrix

	for sample = 1, sampleSize, 1 do

		local randomIndex = Random.new():NextInteger(1, numberOfData)

		table.insert(noiseFeatureMatrixBatch, noiseFeatureMatrix[randomIndex])

	end

	return noiseFeatureMatrixBatch

end

function WassersteinGenerativeAdversarialNetworkModel.new(maxNumberOfIterations, sampleSize)
	
	local NewWassersteinGenerativeAdversarialNetworkModel = {}
	
	setmetatable(NewWassersteinGenerativeAdversarialNetworkModel, WassersteinGenerativeAdversarialNetworkModel)
	
	NewWassersteinGenerativeAdversarialNetworkModel.maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations
	
	NewWassersteinGenerativeAdversarialNetworkModel.sampleSize = sampleSize or defaultSampleSize
	
	NewWassersteinGenerativeAdversarialNetworkModel.isOutputPrinted = true
	
	NewWassersteinGenerativeAdversarialNetworkModel.Generator = nil
	
	NewWassersteinGenerativeAdversarialNetworkModel.Discriminator = nil
	
	return NewWassersteinGenerativeAdversarialNetworkModel
	
end

function WassersteinGenerativeAdversarialNetworkModel:setParameters(maxNumberOfIterations, sampleSize)
	
	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations
	
	self.sampleSize = sampleSize or self.sampleSize
	
end

function WassersteinGenerativeAdversarialNetworkModel:setDiscriminator(Discriminator)
	
	self.Discriminator = Discriminator
	
end

function WassersteinGenerativeAdversarialNetworkModel:setGenerator(Generator)
	
	self.Generator = Generator
	
end

function WassersteinGenerativeAdversarialNetworkModel:setPrintOutput(option)
	
	if (option == false) then

		self.isOutputPrinted = false

	else

		self.isOutputPrinted = true

	end
	
end

function WassersteinGenerativeAdversarialNetworkModel:train(realFeatureMatrix, noiseFeatureMatrix)
	
	local Discriminator = self.Discriminator
	
	local Generator = self.Generator
	
	if (not Discriminator) then error("No discriminator neural network.") end
	
	if (not Generator) then error("No generator neural network.") end
	
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
	
	if (generatorOutputNumberOfFeatures ~= discriminatorInputNumberOfFeatures) then error("The generator's output layer and the discriminator's input layer must contain the same number of neurons.") end
	
	if (discriminatorOutputNumberOfFeatures ~= 1) then error("The number of neurons at the discriminator's output layer must be equal to 1.") end
	
	if (#realFeatureMatrix ~= #noiseFeatureMatrix) then error("Both feature matrices must contain same number of data.") end
	
	if (#noiseFeatureMatrix[1] ~= generatorInputNumberOfFeatures) then error("The number of columns in noise feature matrix must contain the same number as the number of neurons in generator's input layer.") end
	
	if (#realFeatureMatrix[1] ~= discriminatorInputNumberOfFeatures) then error("The number of columns in real feature matrix must contain the same number as the number of neurons in discriminator's input layer.") end

	local discriminatorInputMatrix = AqwamMatrixLibrary:createMatrix(1, discriminatorInputNumberOfFeatures, 1)

	local generatorInputMatrix = AqwamMatrixLibrary:createMatrix(1, generatorInputNumberOfFeatures, 1)
	
	local functionToApplyToDiscriminator = function (discriminatorRealLabel, discriminatorGeneratedLabel) return -(discriminatorRealLabel - discriminatorGeneratedLabel) end
	
	local numberOfIterations = 0
	
	local maxNumberOfIterations = self.maxNumberOfIterations
	
	local sampleSize = self.sampleSize
	
	local isOutputPrinted = self.isOutputPrinted

	repeat
		
		task.wait()
		
		local realFeatureMatrixBatch, noiseFeatureMatrixBatch = samplePair(realFeatureMatrix, noiseFeatureMatrix, sampleSize)
		
		local generatedLabelMatrix = Generator:predict(noiseFeatureMatrix, true)
		
		local discriminatorGeneratedLabelMatrix = Discriminator:predict(generatedLabelMatrix, true)
		
		local discriminatorRealLabelMatrix = Discriminator:predict(realFeatureMatrix, true)
		
		local meanDiscriminatorGeneratedLabelMatrix = AqwamMatrixLibrary:verticalMean(discriminatorGeneratedLabelMatrix)
		
		local meanDiscriminatorRealLabelMatrix = AqwamMatrixLibrary:verticalMean(discriminatorRealLabelMatrix)
		
		local discriminatorLossMatrix = AqwamMatrixLibrary:applyFunction(functionToApplyToDiscriminator, meanDiscriminatorRealLabelMatrix, meanDiscriminatorGeneratedLabelMatrix)
		
		Discriminator:forwardPropagate(discriminatorInputMatrix, true)
		
		Discriminator:backPropagate(discriminatorLossMatrix, true)
		
		numberOfIterations = numberOfIterations + 1
		
		if (isOutputPrinted) then print("Iteration: " .. numberOfIterations .. "\t\tDiscriminator Cost: " .. discriminatorLossMatrix[1][1]) end
		
	until (numberOfIterations >= maxNumberOfIterations)
	
	local finalNoiseFeatureMatrixBatch = sample(noiseFeatureMatrix, sampleSize)
	
	local generatorLossMatrix = Generator:predict(finalNoiseFeatureMatrixBatch, true)
	
	generatorLossMatrix = AqwamMatrixLibrary:multiply(-1, generatorLossMatrix)
	
	local meanGeneratorLossVector = AqwamMatrixLibrary:verticalMean(generatorLossMatrix)
	
	Generator:forwardPropagate(generatorInputMatrix, true)

	Generator:backPropagate(meanGeneratorLossVector, true)
	
end

function WassersteinGenerativeAdversarialNetworkModel:evaluate(featureMatrix)
	
	return self.Discriminator:predict(featureMatrix, true)
	
end

function WassersteinGenerativeAdversarialNetworkModel:generate(noiseFeatureMatrix)
	
	return self.Generator:predict(noiseFeatureMatrix, true)
	
end

function WassersteinGenerativeAdversarialNetworkModel:getGenerator()

	return self.Generator

end

function WassersteinGenerativeAdversarialNetworkModel:getDiscriminator()

	return self.Discriminator

end

return WassersteinGenerativeAdversarialNetworkModel
