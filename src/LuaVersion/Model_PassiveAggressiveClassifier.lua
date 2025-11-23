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

local IterativeMethodBaseModel = require("Model_IterativeMethodBaseModel")

local PassiveAggressiveClassifierModel = {}

PassiveAggressiveClassifierModel.__index = PassiveAggressiveClassifierModel

setmetatable(PassiveAggressiveClassifierModel, IterativeMethodBaseModel)

local defaultMaximumNumberOfIterations = 500

local defaultVariant = "0"

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

function PassiveAggressiveClassifierModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	local NewPassiveAggressiveClassifierModel = IterativeMethodBaseModel.new(parameterDictionary)

	setmetatable(NewPassiveAggressiveClassifierModel, PassiveAggressiveClassifierModel)
	
	NewPassiveAggressiveClassifierModel:setName("PassiveAggressiveClassifier")
	
	NewPassiveAggressiveClassifierModel.variant = parameterDictionary.variant or defaultVariant

	NewPassiveAggressiveClassifierModel.cValue = parameterDictionary.cValue or defaultCValue

	return NewPassiveAggressiveClassifierModel

end

function PassiveAggressiveClassifierModel:train(featureMatrix, labelVector)
	
	local numberOfData = #featureMatrix
	
	if (numberOfData ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end
	
	local ModelParameters = self.ModelParameters

	if (ModelParameters) then

		if (#featureMatrix[1] ~= #ModelParameters) then error("The number of features are not the same as the model parameters.") end

	else

		ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})

	end
	
	local tauFunction = tauFunctionList[self.variant]
	
	if (not tauFunction) then error("Unknown variant.") end
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
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
	
	local tau
	
	local weightChangeVector
	
	local cost
	
	repeat
		
		numberOfIterations = numberOfIterations + 1
		
		self:iterationWait()
		
		totalLoss = 0
		
		for dataIndex, unwrappedFeatureVector in ipairs(featureMatrix) do

			featureVector = {unwrappedFeatureVector}

			labelValue = labelVector[dataIndex][1]

			predictedLabelValue = AqwamTensorLibrary:dotProduct(featureVector, ModelParameters)[1][1]

			transposedFeatureVector = AqwamTensorLibrary:transpose(featureVector)

			dotProductFeatureVectorValue = AqwamTensorLibrary:dotProduct(featureVector, transposedFeatureVector)[1][1]

			lossValue = math.max(0, 1 - (labelValue * predictedLabelValue))

			tau = tauFunction(lossValue, dotProductFeatureVectorValue, cValue)

			weightChangeVector = AqwamTensorLibrary:multiply((tau * labelValue), transposedFeatureVector)

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

function PassiveAggressiveClassifierModel:predict(featureMatrix, returnOriginalOutput)
	
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

return PassiveAggressiveClassifierModel
