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

local AqwamTensorLibrary = require("AqwamTensorLibrary")

local GenerativeAdversarialNetworkBaseModel = require("Model_GenerativeAdversarialNetworkBaseModel")

GenerativeAdversarialNetworkModel = {}

GenerativeAdversarialNetworkModel.__index = GenerativeAdversarialNetworkModel

setmetatable(GenerativeAdversarialNetworkModel, GenerativeAdversarialNetworkBaseModel)

function GenerativeAdversarialNetworkModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewGenerativeAdversarialNetworkModel = GenerativeAdversarialNetworkBaseModel.new(parameterDictionary)
	
	setmetatable(NewGenerativeAdversarialNetworkModel, GenerativeAdversarialNetworkModel)
	
	NewGenerativeAdversarialNetworkModel:setName("GenerativeAdversarialNetwork")
	
	NewGenerativeAdversarialNetworkModel.GeneratorModel = parameterDictionary.GeneratorModel
	
	NewGenerativeAdversarialNetworkModel.DiscriminatorModel = parameterDictionary.GeneratorModel
	
	return NewGenerativeAdversarialNetworkModel
	
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

	local discriminatorInputMatrix = AqwamTensorLibraryLinker:createTensor({1, discriminatorInputNumberOfFeatures}, 1)

	local generatorInputMatrix = AqwamTensorLibraryLinker:createTensor({1, generatorInputNumberOfFeatures}, 1)
	
	local functionToApplyToDiscriminator = function (discriminatorRealLabel, discriminatorGeneratedLabel) return -(math.log(discriminatorRealLabel) + math.log(1 - discriminatorGeneratedLabel)) end
	
	local functionToApplyToGenerator = function (discriminatorGeneratedLabel) return math.log(1 - discriminatorGeneratedLabel) end
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local isOutputPrinted = self.isOutputPrinted
	
	local numberOfIterations = 0
	
	local meanDiscriminatorLossValue = 0

	repeat
		
		task.wait()
		
		local generatedLabelMatrix = GeneratorModel:predict(noiseFeatureMatrix, true)
		
		local discriminatorGeneratedLabelMatrix = DiscriminatorModel:predict(generatedLabelMatrix, true)
		
		local discriminatorRealLabelMatrix = DiscriminatorModel:predict(realFeatureMatrix, true)
		
		local discriminatorLossMatrix = AqwamTensorLibraryLinker:applyFunction(functionToApplyToDiscriminator, discriminatorRealLabelMatrix, discriminatorGeneratedLabelMatrix)
		
		local generatorLossMatrix = AqwamTensorLibraryLinker:applyFunction(functionToApplyToGenerator, discriminatorGeneratedLabelMatrix)
		
		local meanDiscriminatorLossMatrix = AqwamTensorLibraryLinker:mean(discriminatorLossMatrix, 1)
		
		local meanGeneratorLossMatrix = AqwamTensorLibraryLinker:mean(generatorLossMatrix, 1)
		
		meanGeneratorLossMatrix = AqwamTensorLibraryLinker:createTensor({1, generatorOutputNumberOfFeatures}, meanGeneratorLossMatrix[1][1])
		
		DiscriminatorModel:forwardPropagate(discriminatorInputMatrix, true)
		
		DiscriminatorModel:update(meanDiscriminatorLossMatrix, true)
		
		GeneratorModel:forwardPropagate(generatorInputMatrix, true)
		
		GeneratorModel:update(meanGeneratorLossMatrix, true)
		
		numberOfIterations = numberOfIterations + 1
		
		meanDiscriminatorLossValue =  meanDiscriminatorLossMatrix[1][1]
		
		if (isOutputPrinted) then print("Iteration: " .. numberOfIterations .. "\t\tDiscriminator Cost: " .. meanDiscriminatorLossValue) end
		
	until (numberOfIterations >= maximumNumberOfIterations) or self:checkIfTargetCostReached(meanDiscriminatorLossValue) or self:checkIfConverged(meanDiscriminatorLossValue) 
	
end

function GenerativeAdversarialNetworkModel:evaluate(featureMatrix)
	
	local DiscriminatorModel = self.DiscriminatorModel
	
	if (not DiscriminatorModel) then error("No discriminator neural network.") end
	
	return DiscriminatorModel:predict(featureMatrix, true)
	
end

function GenerativeAdversarialNetworkModel:generate(noiseFeatureMatrix)
	
	local GeneratorModel =  self.GeneratorModel
	
	if (not GeneratorModel) then error("No generator neural network.") end
	
	return GeneratorModel:predict(noiseFeatureMatrix, true)
	
end

return GenerativeAdversarialNetworkModel
