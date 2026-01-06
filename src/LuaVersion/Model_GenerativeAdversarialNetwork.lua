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

local GenerativeAdversarialNetworkModel = {}

GenerativeAdversarialNetworkModel.__index = GenerativeAdversarialNetworkModel

setmetatable(GenerativeAdversarialNetworkModel, GenerativeAdversarialNetworkBaseModel)

local discriminatorRealLossGradientFunction = function (discriminatorRealLabel) return (1 / discriminatorRealLabel) end

local discriminatorGeneratedLossGradientFunction = function (discriminatorGeneratedLabel) return (1 / (1 - discriminatorGeneratedLabel)) end

local generatorLossGradientFunction = function (discriminatorGeneratedLabel) return (1 / (1 - discriminatorGeneratedLabel)) end

local discriminatorLossFunction = function (discriminatorRealLabel, discriminatorGeneratedLabel) return (math.log(discriminatorRealLabel) + math.log(1 - discriminatorGeneratedLabel)) end

local function calculateCost(discriminatorRealLabelMatrix, discriminatorGeneratedLabelMatrix)
	
	local lossMatrix = AqwamTensorLibrary:applyFunction(discriminatorLossFunction, discriminatorRealLabelMatrix, discriminatorGeneratedLabelMatrix)
	
	return AqwamTensorLibrary:mean(lossMatrix)
	
end

function GenerativeAdversarialNetworkModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewGenerativeAdversarialNetworkModel = GenerativeAdversarialNetworkBaseModel.new(parameterDictionary)
	
	setmetatable(NewGenerativeAdversarialNetworkModel, GenerativeAdversarialNetworkModel)
	
	NewGenerativeAdversarialNetworkModel:setName("GenerativeAdversarialNetwork")
	
	return NewGenerativeAdversarialNetworkModel
	
end

function GenerativeAdversarialNetworkModel:train(realFeatureMatrix, noiseFeatureMatrix)
	
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
	
	if (#realFeatureMatrix ~= #noiseFeatureMatrix) then error("Both feature matrices must contain same number of data.") end
	
	if (#noiseFeatureMatrix[1] ~= generatorInputNumberOfFeatures) then error("The number of columns in noise feature matrix must contain the same number as the number of neurons in generator's input layer.") end
	
	if (#realFeatureMatrix[1] ~= discriminatorInputNumberOfFeatures) then error("The number of columns in real feature matrix must contain the same number as the number of neurons in discriminator's input layer.") end
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local isOutputPrinted = self.isOutputPrinted
	
	local costArray = {}
	
	local numberOfIterations = 0
	
	local discriminatorCost

	repeat
		
		task.wait()
		
		numberOfIterations = numberOfIterations + 1
		
		local discriminatorRealLabelMatrix = DiscriminatorModel:forwardPropagate(realFeatureMatrix, true)

		local discriminatorRealLossGradientMatrix = AqwamTensorLibrary:applyFunction(discriminatorRealLossGradientFunction, discriminatorRealLabelMatrix)

		DiscriminatorModel:update(discriminatorRealLossGradientMatrix, true)
		
		local generatedLabelMatrix = GeneratorModel:forwardPropagate(noiseFeatureMatrix, true)
		
		local discriminatorGeneratedLabelMatrix = DiscriminatorModel:forwardPropagate(generatedLabelMatrix, true)
		
		local discriminatorGeneratedLossGradientMatrix = AqwamTensorLibrary:applyFunction(discriminatorGeneratedLossGradientFunction, discriminatorGeneratedLabelMatrix)
		
		DiscriminatorModel:update(discriminatorGeneratedLossGradientMatrix, true)
		
		local generatorLossGradientMatrix = AqwamTensorLibrary:applyFunction(generatorLossGradientFunction, discriminatorGeneratedLabelMatrix)
		
		GeneratorModel:update(generatorLossGradientMatrix, true)
		
		discriminatorCost = calculateCost(discriminatorRealLabelMatrix, discriminatorGeneratedLabelMatrix)
		
		table.insert(costArray, discriminatorCost)
		
		if (isOutputPrinted) then print("Iteration: " .. numberOfIterations .. "\t\tDiscriminator Cost: " .. discriminatorCost) end
		
	until (numberOfIterations >= maximumNumberOfIterations) or self:checkIfTargetCostReached(discriminatorCost) or self:checkIfConverged(discriminatorCost)
	
	if (isOutputPrinted) then

		if (discriminatorCost == math.huge) then warn("The model diverged.") end

		if (discriminatorCost ~= discriminatorCost) then warn("The model produced nan (not a number) values.") end

	end
	
	return costArray
	
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
