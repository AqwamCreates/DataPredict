--[[

	--------------------------------------------------------------------

	Aqwam's Machine, Deep And Reinforcement Learning Library (DataPredict)

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

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local GenerativeAdversarialNetworkBaseModel = require(script.Parent.GenerativeAdversarialNetworkBaseModel)

local WassersteinGenerativeAdversarialNetworkModel = {}

WassersteinGenerativeAdversarialNetworkModel.__index = WassersteinGenerativeAdversarialNetworkModel

setmetatable(WassersteinGenerativeAdversarialNetworkModel, GenerativeAdversarialNetworkBaseModel)

local defaultGeneratorMaximumNumberOfIterations = 50

local defaultDiscriminatorMaximumNumberOfIterations = 100

local defaultSampleSize = 3

local functionToApplyToDiscriminator = function (discriminatorRealLabel, discriminatorGeneratedLabel) return (discriminatorRealLabel - discriminatorGeneratedLabel) end

local function calculateCost(discriminatorRealLabelMatrix, discriminatorGeneratedLabelMatrix)

	local lossMatrix = AqwamTensorLibrary:applyFunction(functionToApplyToDiscriminator, discriminatorRealLabelMatrix, discriminatorGeneratedLabelMatrix)

	return AqwamTensorLibrary:mean(lossMatrix)

end

local function sample(matrix, sampleSize)

	local matrixBatch = {}

	local numberOfData = #matrix

	for sample = 1, sampleSize, 1 do

		local randomIndex = Random.new():NextInteger(1, numberOfData)

		table.insert(matrixBatch, matrix[randomIndex])

	end

	return matrixBatch

end

function WassersteinGenerativeAdversarialNetworkModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewWassersteinGenerativeAdversarialNetworkModel = GenerativeAdversarialNetworkBaseModel.new(parameterDictionary)
	
	setmetatable(NewWassersteinGenerativeAdversarialNetworkModel, WassersteinGenerativeAdversarialNetworkModel)
	
	NewWassersteinGenerativeAdversarialNetworkModel:setName("WassersteinGenerativeAdversarialNetwork")
	
	NewWassersteinGenerativeAdversarialNetworkModel.generatorMaximumNumberOfIterations = parameterDictionary.generatorMaximumNumberOfIterations or defaultGeneratorMaximumNumberOfIterations

	NewWassersteinGenerativeAdversarialNetworkModel.discriminatorMaximumNumberOfIterations = parameterDictionary.discriminatorMaximumNumberOfIterations or defaultDiscriminatorMaximumNumberOfIterations
	
	NewWassersteinGenerativeAdversarialNetworkModel.sampleSize = parameterDictionary.sampleSize or defaultSampleSize
	
	return NewWassersteinGenerativeAdversarialNetworkModel
	
end

function WassersteinGenerativeAdversarialNetworkModel:train(realFeatureMatrix, noiseFeatureMatrix)
	
	local DiscriminatorModel = self.DiscriminatorModel
	
	local GeneratorModel = self.GeneratorModel
	
	if (not DiscriminatorModel) then error("No discriminator neural network.") end
	
	if (not GeneratorModel) then error("No generator neural network.") end
	
	local discriminatorNumberOfLayers = DiscriminatorModel:getNumberOfLayers()

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
	
	local generatorNumberOfIterations = 0

	local discriminatorNumberOfIterations = 0

	local generatorMaximumNumberOfIterations = self.generatorMaximumNumberOfIterations

	local discriminatorMaximumNumberOfIterations = self.discriminatorMaximumNumberOfIterations
	
	local sampleSize = self.sampleSize
	
	local isOutputPrinted = self.isOutputPrinted
	
	local discriminatorCost = 0
	
	repeat
		
		repeat

			task.wait()

			local realFeatureMatrixBatch = sample(realFeatureMatrix, sampleSize)

			local noiseFeatureMatrixBatch = sample(noiseFeatureMatrix, sampleSize)
			
			local discriminatorRealLabelMatrix = DiscriminatorModel:forwardPropagate(realFeatureMatrixBatch, true)
			
			local discriminatorRealLossGradientMatrix = AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(discriminatorRealLabelMatrix), 1)
			
			DiscriminatorModel:update(discriminatorRealLossGradientMatrix, true)
			
			local generatedLabelMatrixBatch = GeneratorModel:forwardPropagate(noiseFeatureMatrixBatch, false)

			local discriminatorGeneratedLabelMatrix = DiscriminatorModel:forwardPropagate(generatedLabelMatrixBatch, true)
			
			local discriminatorGeneratedLossGradientMatrix = AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(discriminatorGeneratedLabelMatrix), -1)
			
			DiscriminatorModel:update(discriminatorGeneratedLossGradientMatrix, true)
			
			discriminatorCost = calculateCost(discriminatorRealLabelMatrix, discriminatorGeneratedLabelMatrix)
			
			discriminatorNumberOfIterations = discriminatorNumberOfIterations + 1

			if (isOutputPrinted) then print("Generator Iteration: " .. generatorNumberOfIterations .."\t\tDiscriminator Iteration: " .. discriminatorMaximumNumberOfIterations .. "\t\tDiscriminator Cost: " .. discriminatorCost) end

		until (discriminatorNumberOfIterations >= discriminatorMaximumNumberOfIterations) or self:checkIfTargetCostReached(discriminatorCost) or self:checkIfConverged(discriminatorCost) 

		local finalNoiseFeatureMatrixBatch = sample(noiseFeatureMatrix, sampleSize)

		local finalGeneratedLabelMatrix = GeneratorModel:forwardPropagate(finalNoiseFeatureMatrixBatch, true)

		local generatorLossGradientMatrix = DiscriminatorModel:predict(finalGeneratedLabelMatrix, true)
		
		generatorLossGradientMatrix = AqwamTensorLibrary:unaryMinus(generatorLossGradientMatrix)

		GeneratorModel:update(generatorLossGradientMatrix, true)
		
		generatorNumberOfIterations = generatorNumberOfIterations + 1
		
	until (generatorNumberOfIterations >= generatorMaximumNumberOfIterations)
	
end

function WassersteinGenerativeAdversarialNetworkModel:evaluate(featureMatrix)

	local DiscriminatorModel = self.DiscriminatorModel

	if (not DiscriminatorModel) then error("No discriminator neural network.") end

	return DiscriminatorModel:predict(featureMatrix, true)

end

function WassersteinGenerativeAdversarialNetworkModel:generate(noiseFeatureMatrix)

	local GeneratorModel =  self.GeneratorModel

	if (not GeneratorModel) then error("No generator neural network.") end

	return GeneratorModel:predict(noiseFeatureMatrix, true)

end

return WassersteinGenerativeAdversarialNetworkModel
