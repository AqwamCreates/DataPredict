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

local IterativeMethodBaseModel = require(script.Parent.IterativeMethodBaseModel)

OneClassPassiveAggressiveClassifierModel = {}

OneClassPassiveAggressiveClassifierModel.__index = OneClassPassiveAggressiveClassifierModel

setmetatable(OneClassPassiveAggressiveClassifierModel, IterativeMethodBaseModel)

local defaultMaximumNumberOfIterations = 500

local defaultVariant = "0"

local defaultEpsilon = 0

local defaultCValue = 1

local cutOffFunction = function (x) 

	return ((x > 0) and 1) or ((x < 0) and -1) or 0

end

local tauFunctionList = {

	["0"] = function(lossValue, dotProductFeatureVectorValue, cValue)

		return (lossValue / dotProductFeatureVectorValue)

	end,

	["1"] = function(lossValue, dotProductFeatureVectorValue, cValue)

		return math.min(cValue, (lossValue / dotProductFeatureVectorValue))

	end,

	["2"] = function(lossValue, dotProductFeatureVectorValue, cValue)

		local denominatorValuePart1 = 1 / (2 * cValue)

		return (lossValue / (dotProductFeatureVectorValue + denominatorValuePart1))

	end,

}

function OneClassPassiveAggressiveClassifierModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	local NewOneClassPassiveAggressiveClassifierModel = IterativeMethodBaseModel.new(parameterDictionary)

	setmetatable(NewOneClassPassiveAggressiveClassifierModel, OneClassPassiveAggressiveClassifierModel)

	NewOneClassPassiveAggressiveClassifierModel:setName("OneClassPassiveAggressiveClassifier")

	NewOneClassPassiveAggressiveClassifierModel.variant = parameterDictionary.variant or defaultVariant
	
	NewOneClassPassiveAggressiveClassifierModel.epsilon = parameterDictionary.epsilon or defaultEpsilon

	NewOneClassPassiveAggressiveClassifierModel.cValue = parameterDictionary.cValue or defaultCValue

	return NewOneClassPassiveAggressiveClassifierModel

end

function OneClassPassiveAggressiveClassifierModel:train(featureMatrix, labelVector)
	
	local numberOfData = #featureMatrix
	
	if (labelVector) then
		
		if (numberOfData ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows!") end
		
	else
		
		labelVector = AqwamTensorLibrary:createTensor({numberOfData, 1}, 1)
		
	end
	
	local ModelParameters = self.ModelParameters

	if (ModelParameters) then

		if (#featureMatrix[1] ~= #ModelParameters) then error("The number of features are not the same as the model parameters!") end

	else

		ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})

	end

	local tauFunction = tauFunctionList[self.variant]

	if (not tauFunction) then error("Unknown variant.") end

	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local epsilon = self.epsilon

	local cValue = self.cValue

	local costArray = {}
	
	local numberOfIterations = 0

	local totalLoss

	local featureVector

	local labelValue

	local predictedLabelValue

	local lossValue

	local transposedFeatureVector

	local dotProductFeatureVectorValue
	
	local labelValueSubtractedByWeightVector
	
	local transposedLabelValueSubtractedByWeightVector
	
	local dotProductLabelValueSubtractedByWeightVector
	
	local differenceValue

	local tau
	
	local weightChangeVectorPart1

	local weightChangeVector

	local cost
	
	repeat

		numberOfIterations = numberOfIterations + 1

		self:iterationWait()

		totalLoss = 0
		
		for dataIndex, unwrappedFeatureVector in ipairs(featureMatrix) do
			
			featureVector = {unwrappedFeatureVector}

			labelValue = labelVector[dataIndex][1]

			predictedLabelValue = AqwamTensorLibrary:dotProduct(featureVector, ModelParameters)

			transposedFeatureVector = AqwamTensorLibrary:transpose(featureVector)

			dotProductFeatureVectorValue = AqwamTensorLibrary:dotProduct(featureVector, transposedFeatureVector)

			labelValueSubtractedByWeightVector = AqwamTensorLibrary:subtract(labelValue, ModelParameters)

			transposedLabelValueSubtractedByWeightVector = AqwamTensorLibrary:transpose(labelValueSubtractedByWeightVector)

			dotProductLabelValueSubtractedByWeightVector = AqwamTensorLibrary:dotProduct(transposedLabelValueSubtractedByWeightVector, labelValueSubtractedByWeightVector)

			differenceValue = labelValue - predictedLabelValue

			lossValue = math.max(0, (math.abs(differenceValue) - epsilon))

			tau = tauFunction(lossValue, dotProductFeatureVectorValue, cValue)

			weightChangeVectorPart1 = AqwamTensorLibrary:divide(labelValueSubtractedByWeightVector, dotProductLabelValueSubtractedByWeightVector)

			weightChangeVector = AqwamTensorLibrary:multiply(tau, weightChangeVectorPart1)

			ModelParameters = AqwamTensorLibrary:add(ModelParameters, weightChangeVector)

			totalLoss = totalLoss + lossValue
			
		end

		cost = self:calculateCostWhenRequired(numberOfIterations, function()

			return (totalLoss / numberOfData)

		end)

		if (cost) then 

			table.insert(costArray, cost)

			self:printNumberOfIterationsAndCost(numberOfIterations, cost)

		end
		
	until (numberOfIterations >= maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)
	
	if (self.isOutputPrinted) then

		if (cost == math.huge) then warn("The model diverged.") end

		if (cost ~= cost) then warn("The model produced nan (not a number) values.") end

	end

	self.ModelParameters = ModelParameters

	return costArray

end

function OneClassPassiveAggressiveClassifierModel:predict(featureMatrix, returnOriginalOutput)

	local ModelParameters = self.ModelParameters

	if (not ModelParameters) then

		ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})

		self.ModelParameters = ModelParameters

	end

	local outputVector = AqwamTensorLibrary:dotProduct(featureMatrix, ModelParameters)
	
	if (returnOriginalOutput) then return outputVector end

	local predictedLabelVector = AqwamTensorLibrary:applyFunction(cutOffFunction, outputVector)

	return predictedLabelVector, outputVector

end

return OneClassPassiveAggressiveClassifierModel
