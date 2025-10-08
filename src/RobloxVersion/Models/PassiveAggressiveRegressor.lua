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

PassiveAggressiveRegressorModel = {}

PassiveAggressiveRegressorModel.__index = PassiveAggressiveRegressorModel

setmetatable(PassiveAggressiveRegressorModel, IterativeMethodBaseModel)

local defaultMaximumNumberOfIterations = math.huge

local defaultVariant = "0"

local defaultEpsilon = 0

local defaultCValue = 1

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

function PassiveAggressiveRegressorModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	local NewPassiveAggressiveRegressorModel = IterativeMethodBaseModel.new(parameterDictionary)

	setmetatable(NewPassiveAggressiveRegressorModel, PassiveAggressiveRegressorModel)

	NewPassiveAggressiveRegressorModel:setName("PassiveAggressiveRegressor")

	NewPassiveAggressiveRegressorModel.variant = parameterDictionary.variant or defaultVariant
	
	NewPassiveAggressiveRegressorModel.epsilon = parameterDictionary.epsilon or defaultEpsilon

	NewPassiveAggressiveRegressorModel.cValue = parameterDictionary.cValue or defaultCValue

	return NewPassiveAggressiveRegressorModel

end

function PassiveAggressiveRegressorModel:train(featureMatrix, labelVector)

	local ModelParameters = self.ModelParameters

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows!") end

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

	local totalLoss = 0

	local featureVector

	local labelValue

	local predictedLabelValue

	local lossValue

	local transposedFeatureVector

	local dotProductFeatureVectorValue
	
	local differenceValue

	local tau

	local weightChangeVector

	local cost

	for dataIndex, unwrappedFeatureVector in ipairs(featureMatrix) do

		featureVector = {unwrappedFeatureVector}

		labelValue = labelVector[dataIndex][1]

		predictedLabelValue = AqwamTensorLibrary:dotProduct(featureVector, ModelParameters)

		transposedFeatureVector = AqwamTensorLibrary:transpose(featureVector)

		dotProductFeatureVectorValue = AqwamTensorLibrary:dotProduct(featureVector, transposedFeatureVector)
		
		differenceValue = labelValue - predictedLabelValue

		lossValue = math.max(0, (math.abs(differenceValue) - epsilon))

		tau = tauFunction(lossValue, dotProductFeatureVectorValue, cValue)

		weightChangeVector = AqwamTensorLibrary:multiply((tau * math.sign(differenceValue)), transposedFeatureVector)

		ModelParameters = AqwamTensorLibrary:add(ModelParameters, weightChangeVector)

		totalLoss = totalLoss + lossValue

		cost = self:calculateCostWhenRequired(dataIndex, function()

			return (totalLoss / dataIndex)

		end)

		if (cost) then 

			table.insert(costArray, cost)

			self:printNumberOfIterationsAndCost(dataIndex, cost)

		end

		if (dataIndex >= maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost) then break end

	end

	if (cost == math.huge) then warn("The model diverged! Please repeat the experiment again or change the argument values.") end

	self.ModelParameters = ModelParameters

	return costArray

end

function PassiveAggressiveRegressorModel:predict(featureMatrix)

	local ModelParameters = self.ModelParameters

	if (not ModelParameters) then error("No model parameters.") end

	local outputVector = AqwamTensorLibrary:dotProduct(featureMatrix, ModelParameters)

	if (type(outputVector) ~= "table") then outputVector = {{outputVector}} end

	return outputVector

end

return PassiveAggressiveRegressorModel
