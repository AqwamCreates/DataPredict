GenerativeAdversarialNetwork = {}

GenerativeAdversarialNetwork.__index = GenerativeAdversarialNetwork

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local defaultMaxNumberOfIterations = 500

function GenerativeAdversarialNetwork.new(maxNumberOfIterations)
	
	local NewGenerativeAdversarialNetwork = {}
	
	setmetatable(NewGenerativeAdversarialNetwork, GenerativeAdversarialNetwork)
	
	NewGenerativeAdversarialNetwork.maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations
	
	NewGenerativeAdversarialNetwork.isOutputPrinted = true
	
	NewGenerativeAdversarialNetwork.GeneratorNeuralNetwork = nil
	
	NewGenerativeAdversarialNetwork.DiscriminatorNeuralNetwork = nil
	
	return NewGenerativeAdversarialNetwork
	
end

function GenerativeAdversarialNetwork:setParameters(maxNumberOfIterations)
	
	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations
	
end

function GenerativeAdversarialNetwork:setDiscriminatorNeuralNetwork(DiscriminatorNeuralNetwork)
	
	self.DiscriminatorNeuralNetwork = DiscriminatorNeuralNetwork
	
end

function GenerativeAdversarialNetwork:setGeneratorNeuralNetwork(GeneratorNeuralNetwork)
	
	self.GeneratorNeuralNetwork = GeneratorNeuralNetwork
	
end

function GenerativeAdversarialNetwork:setPrintOutput(option)
	
	if (option == false) then

		self.isOutputPrinted = false

	else

		self.isOutputPrinted = true

	end
	
end

function GenerativeAdversarialNetwork:train(realFeatureMatrix, noiseFeatureMatrix)
	
	if (#realFeatureMatrix ~= noiseFeatureMatrix) then error("Both feature matrices must contain same number of data.") end
	
	local DiscriminatorNeuralNetwork = self.DiscriminatorNeuralNetwork
	
	local GeneratorNeuralNetwork = self.GeneratorNeuralNetwork
	
	if (not DiscriminatorNeuralNetwork) then error("No discriminator neural network.") end
	
	if (not GeneratorNeuralNetwork) then error("No generator neural network.") end
	
	local discriminatorNumberOfFeatures, discriminatorHasBias = GeneratorNeuralNetwork:getLayer(1)
	
	local generatorNumberOfFeatures, generatorHasBias = GeneratorNeuralNetwork:getLayer(1)
	
	local functionToApplyToDiscriminator = function (discriminatorRealLabel, discriminatorGeneratedLabel) return math.log(discriminatorRealLabel) + math.log(1 - discriminatorGeneratedLabel) end
	
	local functionToApplyToGenerator = function (discriminatorGeneratedLabel) return math.log(1 - discriminatorGeneratedLabel) end
	
	local numberOfIterations = 0
	
	local maxNumberOfIterations = self.maxNumberOfIterations
	
	local isOutputPrinted = self.isOutputPrinted

	discriminatorNumberOfFeatures = discriminatorNumberOfFeatures + (discriminatorHasBias and 1) or 0
	
	generatorNumberOfFeatures = generatorNumberOfFeatures + (generatorHasBias and 1) or 0
	
	local discriminatorInputMatrix = AqwamMatrixLibrary:createMatrix(1, discriminatorNumberOfFeatures, 1)
	
	local generatorInputMatrix = AqwamMatrixLibrary:createMatrix(1, generatorNumberOfFeatures, 1)
	
	repeat
		
		task.wait()
		
		local generatedLabelMatrix = GeneratorNeuralNetwork:predict(noiseFeatureMatrix, true)
		
		local discriminatorGeneratedLabelMatrix = DiscriminatorNeuralNetwork:predict(generatedLabelMatrix, true)
		
		local discriminatorRealLabelMatrix = DiscriminatorNeuralNetwork:predict(generatedLabelMatrix, true)
		
		local discriminatorLossMatrix = AqwamMatrixLibrary:applyFunction(functionToApplyToDiscriminator, discriminatorRealLabelMatrix, discriminatorGeneratedLabelMatrix)
		
		local generatorLossMatrix = AqwamMatrixLibrary:applyFunction(functionToApplyToGenerator, discriminatorGeneratedLabelMatrix)
		
		local meanDiscriminatorLossMatrix = AqwamMatrixLibrary:verticalMean(discriminatorLossMatrix)
		
		local meanGeneratorLossVector = AqwamMatrixLibrary:verticalMean(generatorLossMatrix)
		
		DiscriminatorNeuralNetwork:forwardPropagate(discriminatorInputMatrix, true)
		
		DiscriminatorNeuralNetwork:backPropagate(meanDiscriminatorLossMatrix, true)
		
		GeneratorNeuralNetwork:forwardPropagate(generatorInputMatrix, true)
		
		GeneratorNeuralNetwork:backPropagate(meanGeneratorLossVector, true)
		
		numberOfIterations = numberOfIterations + 1
		
		if isOutputPrinted then print("Iteration: " .. numberOfIterations) end
		
	until (numberOfIterations >= maxNumberOfIterations)
	
end

function GenerativeAdversarialNetwork:evaluate(featureMatrix)
	
	return self.DiscriminatorNeuralNetwork:predict(featureMatrix, true)
	
end

function GenerativeAdversarialNetwork:generate(featureMatrix)
	
	return self.GeneratorNeuralNetwork:predict(featureMatrix, true)
	
end

return GenerativeAdversarialNetwork
