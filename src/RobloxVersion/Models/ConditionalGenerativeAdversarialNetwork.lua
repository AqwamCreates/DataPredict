--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

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

ConditionalGenerativeAdversarialNetworkModel = {}

ConditionalGenerativeAdversarialNetworkModel.__index = ConditionalGenerativeAdversarialNetworkModel

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local defaultMaximumNumberOfIterations = 500

function ConditionalGenerativeAdversarialNetworkModel.new(maximumNumberOfIterations)
	
	local NewConditionalGenerativeAdversarialNetworkModel = {}
	
	setmetatable(NewConditionalGenerativeAdversarialNetworkModel, ConditionalGenerativeAdversarialNetworkModel)
	
	NewConditionalGenerativeAdversarialNetworkModel.maximumNumberOfIterations = maximumNumberOfIterations or defaultMaximumNumberOfIterations
	
	NewConditionalGenerativeAdversarialNetworkModel.isOutputPrinted = true
	
	NewConditionalGenerativeAdversarialNetworkModel.GeneratorModel = nil
	
	NewConditionalGenerativeAdversarialNetworkModel.DiscriminatorModel = nil
	
	return NewConditionalGenerativeAdversarialNetworkModel
	
end

function ConditionalGenerativeAdversarialNetworkModel:setParameters(maximumNumberOfIterations)
	
	self.maximumNumberOfIterations = maximumNumberOfIterations or self.maximumNumberOfIterations
	
end

function ConditionalGenerativeAdversarialNetworkModel:setDiscriminatorModel(DiscriminatorModel)
	
	self.DiscriminatorModel = DiscriminatorModel
	
end

function ConditionalGenerativeAdversarialNetworkModel:setGeneratorModel(GeneratorModel)
	
	self.GeneratorModel = GeneratorModel
	
end

function ConditionalGenerativeAdversarialNetworkModel:setPrintOutput(option)

	self.isOutputPrinted = option

end

function ConditionalGenerativeAdversarialNetworkModel:train(realFeatureMatrix, noiseFeatureMatrix, labelMatrix)
	
	local DiscriminatorModel = self.DiscriminatorModel
	
	local GeneratorModel = self.GeneratorModel
	
	if (not DiscriminatorModel) then error("No discriminator neural network.") end
	
	if (not GeneratorModel) then error("No generator neural network.") end
	
	local numberOfFeaturesInLabelMatrix = #labelMatrix[1]
	
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
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local isOutputPrinted = self.isOutputPrinted

	repeat
		
		task.wait()
		
		local generatedLabelMatrix = GeneratorModel:predict(concatenatedNoiseFeatureMatrix, true)
		
		local concatenatedAndGeneratedLabelMatrix = AqwamMatrixLibrary:horizontalConcatenate(generatedLabelMatrix, labelMatrix)
		
		local discriminatorGeneratedLabelMatrix = DiscriminatorModel:predict(concatenatedAndGeneratedLabelMatrix, true)
		
		local discriminatorRealLabelMatrix = DiscriminatorModel:predict(concatenatedRealFeatureMatrix, true)
		
		local discriminatorLossMatrix = AqwamMatrixLibrary:applyFunction(functionToApplyToDiscriminator, discriminatorRealLabelMatrix, discriminatorGeneratedLabelMatrix)
		
		local generatorLossMatrix = AqwamMatrixLibrary:applyFunction(functionToApplyToGenerator, discriminatorGeneratedLabelMatrix)
		
		local meanDiscriminatorLossMatrix = AqwamMatrixLibrary:verticalMean(discriminatorLossMatrix)
		
		local meanGeneratorLossMatrix = AqwamMatrixLibrary:verticalMean(generatorLossMatrix)
		
		meanGeneratorLossMatrix = AqwamMatrixLibrary:createMatrix(1, generatorOutputNumberOfFeatures, meanGeneratorLossMatrix[1][1])
		
		DiscriminatorModel:forwardPropagate(discriminatorInputMatrix, true)
		
		DiscriminatorModel:backwardPropagate(meanDiscriminatorLossMatrix, true)
		
		GeneratorModel:forwardPropagate(generatorInputMatrix, true)
		
		GeneratorModel:backwardPropagate(meanGeneratorLossMatrix, true)
		
		numberOfIterations = numberOfIterations + 1
		
		if (isOutputPrinted) then print("Iteration: " .. numberOfIterations .. "\t\tDiscriminator Cost: " .. meanDiscriminatorLossMatrix[1][1]) end
		
	until (numberOfIterations >= maximumNumberOfIterations)
	
end

function ConditionalGenerativeAdversarialNetworkModel:evaluate(featureMatrix, labelMatrix)
	
	if (#featureMatrix ~= #labelMatrix) then error("The feature matrix and the label matrix must contain same number of data.") end
	
	local concatenatedMatrices = AqwamMatrixLibrary:horizontalConcatenate(featureMatrix, labelMatrix)
	
	return self.DiscriminatorModel:predict(concatenatedMatrices, true)
	
end

function ConditionalGenerativeAdversarialNetworkModel:generate(noiseFeatureMatrix, labelMatrix)
	
	if (#noiseFeatureMatrix ~= #labelMatrix) then error("The noise feature matrix and the label matrix must contain same number of data.") end

	local concatenatedMatrices = AqwamMatrixLibrary:horizontalConcatenate(noiseFeatureMatrix, labelMatrix)
	
	return self.GeneratorModel:predict(concatenatedMatrices, true)
	
end

function ConditionalGenerativeAdversarialNetworkModel:getDiscriminatorModel()
	
	return self.DiscriminatorModel
	
end

function ConditionalGenerativeAdversarialNetworkModel:getGeneratorModel()

	return self.GeneratorModel

end

return ConditionalGenerativeAdversarialNetworkModel
