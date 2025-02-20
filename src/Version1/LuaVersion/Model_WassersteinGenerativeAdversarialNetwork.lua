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

WassersteinGenerativeAdversarialNetworkModel = {}

WassersteinGenerativeAdversarialNetworkModel.__index = WassersteinGenerativeAdversarialNetworkModel

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

local defaultGeneratorMaximumNumberOfIterations = 50

local defaultDiscriminatorMaximumNumberOfIterations = 100

local defaultSampleSize = 3

local function sample(matrix, sampleSize)

	local matrixBatch = {}

	local numberOfData = #matrix

	for sample = 1, sampleSize, 1 do

		local randomIndex = Random.new():NextInteger(1, numberOfData)

		table.insert(matrixBatch, matrix[randomIndex])

	end

	return matrixBatch

end

function WassersteinGenerativeAdversarialNetworkModel.new(generatorMaximumNumberOfIterations, discriminatorMaximumNumberOfIterations, sampleSize)
	
	local NewWassersteinGenerativeAdversarialNetworkModel = {}
	
	setmetatable(NewWassersteinGenerativeAdversarialNetworkModel, WassersteinGenerativeAdversarialNetworkModel)
	
	NewWassersteinGenerativeAdversarialNetworkModel.generatorMaximumNumberOfIterations = generatorMaximumNumberOfIterations or defaultGeneratorMaximumNumberOfIterations

	NewWassersteinGenerativeAdversarialNetworkModel.discriminatorMaximumNumberOfIterations = discriminatorMaximumNumberOfIterations or defaultDiscriminatorMaximumNumberOfIterations
	
	NewWassersteinGenerativeAdversarialNetworkModel.sampleSize = sampleSize or defaultSampleSize
	
	NewWassersteinGenerativeAdversarialNetworkModel.isOutputPrinted = true
	
	NewWassersteinGenerativeAdversarialNetworkModel.GeneratorModel = nil
	
	NewWassersteinGenerativeAdversarialNetworkModel.DiscriminatorModel = nil
	
	return NewWassersteinGenerativeAdversarialNetworkModel
	
end

function WassersteinGenerativeAdversarialNetworkModel:setParameters(generatorMaximumNumberOfIterations, discriminatorMaximumNumberOfIterations, sampleSize)
	
	self.generatorMaximumNumberOfIterations = generatorMaximumNumberOfIterations or self.generatorMaximumNumberOfIterations

	self.discriminatorMaximumNumberOfIterations = discriminatorMaximumNumberOfIterations or self.discriminatorMaximumNumberOfIterations

	self.sampleSize = sampleSize or self.sampleSize
	
end

function WassersteinGenerativeAdversarialNetworkModel:setDiscriminatorModel(DiscriminatorModel)
	
	self.DiscriminatorModel = DiscriminatorModel
	
end

function WassersteinGenerativeAdversarialNetworkModel:setGeneratorModel(GeneratorModel)
	
	self.GeneratorModel = GeneratorModel
	
end

function WassersteinGenerativeAdversarialNetworkModel:setPrintOutput(option)
	
	self.isOutputPrinted = option
	
end

function WassersteinGenerativeAdversarialNetworkModel:train(realFeatureMatrix, noiseFeatureMatrix)
	
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
	
	if (#noiseFeatureMatrix[1] ~= generatorInputNumberOfFeatures) then error("The number of columns in noise feature matrix must contain the same number as the number of neurons in generator's input layer.") end
	
	if (#realFeatureMatrix[1] ~= discriminatorInputNumberOfFeatures) then error("The number of columns in real feature matrix must contain the same number as the number of neurons in discriminator's input layer.") end
	
	local discriminatorInputMatrix = AqwamMatrixLibrary:createMatrix(1, discriminatorInputNumberOfFeatures, 1)

	local generatorInputMatrix = AqwamMatrixLibrary:createMatrix(1, generatorInputNumberOfFeatures, 1)
	
	local functionToApplyToDiscriminator = function (discriminatorRealLabel, discriminatorGeneratedLabel) return -(discriminatorRealLabel - discriminatorGeneratedLabel) end
	
	local generatorNumberOfIterations = 0

	local discriminatorNumberOfIterations = 0

	local generatorMaximumNumberOfIterations = self.generatorMaximumNumberOfIterations

	local discriminatorMaximumNumberOfIterations = self.discriminatorMaximumNumberOfIterations
	
	local sampleSize = self.sampleSize
	
	local isOutputPrinted = self.isOutputPrinted
	
	repeat
		
		repeat

			task.wait()

			local realFeatureMatrixBatch = sample(realFeatureMatrix, sampleSize)

			local noiseFeatureMatrixBatch = sample(noiseFeatureMatrix, sampleSize)

			local generatedLabelMatrixBatch = GeneratorModel:predict(noiseFeatureMatrix, true)

			local discriminatorGeneratedLabelMatrix = DiscriminatorModel:predict(generatedLabelMatrixBatch, true)

			local discriminatorRealLabelMatrix = DiscriminatorModel:predict(realFeatureMatrixBatch, true)

			local meanDiscriminatorGeneratedLabelMatrix = AqwamMatrixLibrary:verticalMean(discriminatorGeneratedLabelMatrix)

			local meanDiscriminatorRealLabelMatrix = AqwamMatrixLibrary:verticalMean(discriminatorRealLabelMatrix)

			local discriminatorLossMatrix = AqwamMatrixLibrary:applyFunction(functionToApplyToDiscriminator, meanDiscriminatorRealLabelMatrix, meanDiscriminatorGeneratedLabelMatrix)

			DiscriminatorModel:forwardPropagate(discriminatorInputMatrix, true)

			DiscriminatorModel:backwardPropagate(discriminatorLossMatrix, true)

			discriminatorNumberOfIterations = discriminatorNumberOfIterations + 1

			if (isOutputPrinted) then print("Generator Iteration: " .. generatorNumberOfIterations .."\t\tDiscriminator Iteration: " .. discriminatorMaximumNumberOfIterations .. "\t\tDiscriminator Cost: " .. discriminatorLossMatrix[1][1]) end

		until (discriminatorNumberOfIterations >= generatorMaximumNumberOfIterations)

		local finalNoiseFeatureMatrixBatch = sample(noiseFeatureMatrix, sampleSize)

		local finalGeneratedLabelMatrix = GeneratorModel:predict(finalNoiseFeatureMatrixBatch, true)

		local generatorLossMatrix = DiscriminatorModel:predict(finalGeneratedLabelMatrix, true)

		local meanGeneratorLossVector = AqwamMatrixLibrary:verticalMean(generatorLossMatrix)

		meanGeneratorLossVector = AqwamMatrixLibrary:multiply(-1, meanGeneratorLossVector)

		meanGeneratorLossVector = AqwamMatrixLibrary:createMatrix(1, generatorOutputNumberOfFeatures, meanGeneratorLossVector[1][1])

		GeneratorModel:forwardPropagate(generatorInputMatrix, true)

		GeneratorModel:backwardPropagate(meanGeneratorLossVector, true)
		
		generatorNumberOfIterations = generatorNumberOfIterations + 1
		
	until (generatorNumberOfIterations >= generatorMaximumNumberOfIterations)
	
end

function WassersteinGenerativeAdversarialNetworkModel:evaluate(featureMatrix)
	
	return self.DiscriminatorModel:predict(featureMatrix, true)
	
end

function WassersteinGenerativeAdversarialNetworkModel:generate(noiseFeatureMatrix)
	
	return self.GeneratorModel:predict(noiseFeatureMatrix, true)
	
end

function WassersteinGenerativeAdversarialNetworkModel:getDiscriminatorModel()

	return self.DiscriminatorModel

end

function WassersteinGenerativeAdversarialNetworkModel:getGeneratorModel()

	return self.GeneratorModel

end

return WassersteinGenerativeAdversarialNetworkModel