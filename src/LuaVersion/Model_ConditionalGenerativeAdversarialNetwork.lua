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

local ConditionalGenerativeAdversarialNetworkModel = {}

ConditionalGenerativeAdversarialNetworkModel.__index = ConditionalGenerativeAdversarialNetworkModel

setmetatable(ConditionalGenerativeAdversarialNetworkModel, GenerativeAdversarialNetworkBaseModel)

local discriminatorRealLossGradientFunction = function (discriminatorRealLabel) return (1 / discriminatorRealLabel) end

local discriminatorGeneratedLossGradientFunction = function (discriminatorGeneratedLabel) return (1 / (1 - discriminatorGeneratedLabel)) end

local generatorLossGradientFunction = function (discriminatorGeneratedLabel) return (1 / (1 - discriminatorGeneratedLabel)) end

local discriminatorLossFunction = function (discriminatorRealLabel, discriminatorGeneratedLabel) return (math.log(discriminatorRealLabel) + math.log(1 - discriminatorGeneratedLabel)) end

local function calculateCost(discriminatorRealLabelMatrix, discriminatorGeneratedLabelMatrix)

	local lossMatrix = AqwamTensorLibrary:applyFunction(discriminatorLossFunction, discriminatorRealLabelMatrix, discriminatorGeneratedLabelMatrix)

	return AqwamTensorLibrary:mean(lossMatrix)

end

function ConditionalGenerativeAdversarialNetworkModel.new(parameterDictionary)
	
	local NewConditionalGenerativeAdversarialNetworkModel = GenerativeAdversarialNetworkBaseModel.new(parameterDictionary)
	
	setmetatable(NewConditionalGenerativeAdversarialNetworkModel, ConditionalGenerativeAdversarialNetworkModel)
	
	NewConditionalGenerativeAdversarialNetworkModel:setName("ConditionalGenerativeAdversarialNetwork")
	
	return NewConditionalGenerativeAdversarialNetworkModel
	
end

function ConditionalGenerativeAdversarialNetworkModel:train(realFeatureMatrix, noiseFeatureMatrix, labelMatrix)
	
	local DiscriminatorModel = self.DiscriminatorModel
	
	local GeneratorModel = self.GeneratorModel
	
	if (not DiscriminatorModel) then error("No discriminator neural network.") end
	
	if (not GeneratorModel) then error("No generator neural network.") end
	
	local numberOfFeaturesInLabelMatrix = #labelMatrix[1]
	
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
	
	if ((generatorOutputNumberOfFeatures + numberOfFeaturesInLabelMatrix) ~= discriminatorInputNumberOfFeatures) then error("The number of neurons at the discriminator's input layer must equal to the total of number of neurons at the generator's output layer and the number of features in label matrix.") end
	
	if (discriminatorOutputNumberOfFeatures ~= 1) then error("The number of neurons at the discriminator's output layer must be equal to 1.") end
	
	if (#realFeatureMatrix ~= #noiseFeatureMatrix) or (#realFeatureMatrix ~= #labelMatrix) then error("All matrices must contain same number of data.") end
	
	if ((#noiseFeatureMatrix[1] + numberOfFeaturesInLabelMatrix)  ~= generatorInputNumberOfFeatures) then error("The total number of columns in noise feature matrix and label matrix must contain the same number as the number of neurons in generator's input layer.") end
	
	if ((#realFeatureMatrix[1] + numberOfFeaturesInLabelMatrix) ~= discriminatorInputNumberOfFeatures) then error("The total number of columns in real feature matrix and label matrix must contain the same number as the number of neurons in discriminator's input layer.") end
	
	local concatenatedNoiseFeatureMatrix = AqwamTensorLibrary:concatenate(noiseFeatureMatrix, labelMatrix, 2)
	
	local concatenatedRealFeatureMatrix = AqwamTensorLibrary:concatenate(realFeatureMatrix, labelMatrix, 2)
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local isOutputPrinted = self.isOutputPrinted
	
	local costArray = {}
	
	local numberOfIterations = 0
	
	local discriminatorCost

	repeat
		
		self:iterationWait()
		
		numberOfIterations = numberOfIterations + 1
			
		local discriminatorRealLabelMatrix = DiscriminatorModel:forwardPropagate(concatenatedRealFeatureMatrix, true)

		local discriminatorRealLossGradientMatrix = AqwamTensorLibrary:applyFunction(discriminatorRealLossGradientFunction, discriminatorRealLabelMatrix)

		DiscriminatorModel:update(discriminatorRealLossGradientMatrix, true)

		local generatedLabelMatrix = GeneratorModel:forwardPropagate(concatenatedNoiseFeatureMatrix, true)
		
		local concatenatedGeneratedLabelMatrix = AqwamTensorLibrary:concatenate(generatedLabelMatrix, labelMatrix, 2)

		local discriminatorGeneratedLabelMatrix = DiscriminatorModel:forwardPropagate(concatenatedGeneratedLabelMatrix, true)

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

function ConditionalGenerativeAdversarialNetworkModel:evaluate(featureMatrix, labelMatrix)
	
	local DiscriminatorModel = self.DiscriminatorModel

	if (not DiscriminatorModel) then error("No discriminator neural network.") end
	
	if (#featureMatrix ~= #labelMatrix) then error("The feature matrix and the label matrix must contain same number of data.") end
	
	local concatenatedMatrices = AqwamTensorLibrary:concatenate(featureMatrix, labelMatrix, 2)
	
	return DiscriminatorModel:predict(concatenatedMatrices, true)
	
end

function ConditionalGenerativeAdversarialNetworkModel:generate(noiseFeatureMatrix, labelMatrix)
	
	local GeneratorModel =  self.GeneratorModel

	if (not GeneratorModel) then error("No generator neural network.") end
	
	if (#noiseFeatureMatrix ~= #labelMatrix) then error("The noise feature matrix and the label matrix must contain same number of data.") end

	local concatenatedMatrices = AqwamTensorLibrary:concatenate(noiseFeatureMatrix, labelMatrix, 2)
	
	return GeneratorModel:predict(concatenatedMatrices, true)
	
end

return ConditionalGenerativeAdversarialNetworkModel
