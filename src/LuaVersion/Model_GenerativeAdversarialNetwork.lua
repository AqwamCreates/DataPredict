GenerativeAdversarialNetworkModel = {}

GenerativeAdversarialNetworkModel.__index = GenerativeAdversarialNetworkModel

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

local defaultMaxNumberOfIterations = 500

function GenerativeAdversarialNetworkModel.new(maxNumberOfIterations)
	
	local NewGenerativeAdversarialNetworkModel = {}
	
	setmetatable(NewGenerativeAdversarialNetworkModel, GenerativeAdversarialNetworkModel)
	
	NewGenerativeAdversarialNetworkModel.maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations
	
	NewGenerativeAdversarialNetworkModel.isOutputPrinted = true
	
	NewGenerativeAdversarialNetworkModel.GeneratorModel = nil
	
	NewGenerativeAdversarialNetworkModel.DiscriminatorModel = nil
	
	return NewGenerativeAdversarialNetworkModel
	
end

function GenerativeAdversarialNetworkModel:setParameters(maxNumberOfIterations)
	
	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations
	
end

function GenerativeAdversarialNetworkModel:setDiscriminator(DiscriminatorModel)
	
	self.DiscriminatorModel = DiscriminatorModel
	
end

function GenerativeAdversarialNetworkModel:setGeneratorModel(GeneratorModel)
	
	self.GeneratorModel = GeneratorModel
	
end

function GenerativeAdversarialNetworkModel:setPrintOutput(option)
	
	if (option == false) then

		self.isOutputPrinted = false

	else

		self.isOutputPrinted = true

	end
	
end

function GenerativeAdversarialNetworkModel:train(realFeatureMatrix, noiseFeatureMatrix)
	
	local DiscriminatorModel = self.DiscriminatorModel
	
	local GeneratorModel = self.GeneratorModel
	
	if (not DiscriminatorModel) then error("No discriminator neural network.") end
	
	if (not GeneratorModel) then error("No generator neural network.") end
	
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
	
	if (generatorOutputNumberOfFeatures ~= discriminatorInputNumberOfFeatures) then error("The generator's output layer and the discriminator's input layer must contain the same number of neurons.") end
	
	if (discriminatorOutputNumberOfFeatures ~= 1) then error("The number of neurons at the discriminator's output layer must be equal to 1.") end
	
	if (#realFeatureMatrix ~= #noiseFeatureMatrix) then error("Both feature matrices must contain same number of data.") end
	
	if (#noiseFeatureMatrix[1] ~= generatorInputNumberOfFeatures) then error("The number of columns in noise feature matrix must contain the same number as the number of neurons in generator's input layer.") end
	
	if (#realFeatureMatrix[1] ~= discriminatorInputNumberOfFeatures) then error("The number of columns in real feature matrix must contain the same number as the number of neurons in discriminator's input layer.") end

	local discriminatorInputMatrix = AqwamMatrixLibrary:createMatrix(1, discriminatorInputNumberOfFeatures, 1)

	local generatorInputMatrix = AqwamMatrixLibrary:createMatrix(1, generatorInputNumberOfFeatures, 1)
	
	local functionToApplyToDiscriminator = function (discriminatorRealLabel, discriminatorGeneratedLabel) return -(math.log(discriminatorRealLabel) + math.log(1 - discriminatorGeneratedLabel)) end
	
	local functionToApplyToGenerator = function (discriminatorGeneratedLabel) return math.log(1 - discriminatorGeneratedLabel) end
	
	local numberOfIterations = 0
	
	local maxNumberOfIterations = self.maxNumberOfIterations
	
	local isOutputPrinted = self.isOutputPrinted

	repeat
		
		task.wait()
		
		local generatedLabelMatrix = GeneratorModel:predict(noiseFeatureMatrix, true)
		
		local discriminatorGeneratedLabelMatrix = DiscriminatorModel:predict(generatedLabelMatrix, true)
		
		local discriminatorRealLabelMatrix = DiscriminatorModel:predict(realFeatureMatrix, true)
		
		local discriminatorLossMatrix = AqwamMatrixLibrary:applyFunction(functionToApplyToDiscriminator, discriminatorRealLabelMatrix, discriminatorGeneratedLabelMatrix)
		
		local generatorLossMatrix = AqwamMatrixLibrary:applyFunction(functionToApplyToGenerator, discriminatorGeneratedLabelMatrix)
		
		local meanDiscriminatorLossMatrix = AqwamMatrixLibrary:verticalMean(discriminatorLossMatrix)
		
		local meanGeneratorLossVector = AqwamMatrixLibrary:verticalMean(generatorLossMatrix)
		
		meanGeneratorLossVector = AqwamMatrixLibrary:createMatrix(1, generatorOutputNumberOfFeatures, meanGeneratorLossVector[1][1])
		
		DiscriminatorModel:forwardPropagate(discriminatorInputMatrix, true)
		
		DiscriminatorModel:backPropagate(meanDiscriminatorLossMatrix, true)
		
		GeneratorModel:forwardPropagate(generatorInputMatrix, true)
		
		GeneratorModel:backPropagate(meanGeneratorLossVector, true)
		
		numberOfIterations = numberOfIterations + 1
		
		if (isOutputPrinted) then print("Iteration: " .. numberOfIterations .. "\t\tDiscriminator Cost: " .. meanDiscriminatorLossMatrix[1][1]) end
		
	until (numberOfIterations >= maxNumberOfIterations)
	
end

function GenerativeAdversarialNetworkModel:evaluate(featureMatrix)
	
	return self.DiscriminatorModel:predict(featureMatrix, true)
	
end

function GenerativeAdversarialNetworkModel:generate(noiseFeatureMatrix)
	
	return self.GeneratorModel:predict(noiseFeatureMatrix, true)
	
end

function GenerativeAdversarialNetworkModel:getGeneratorModel()

	return self.GeneratorModel

end

function GenerativeAdversarialNetworkModel:getDiscriminatorModel()

	return self.DiscriminatorModel

end

return GenerativeAdversarialNetworkModel
