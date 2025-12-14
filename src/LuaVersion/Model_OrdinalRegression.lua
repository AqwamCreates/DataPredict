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

local GradientMethodBaseModel = require("Model_GradientMethodBaseModel")

local ZTableFunction = require("Core_ZTableFunction")

local OrdinalRegressionModel = {}

OrdinalRegressionModel.__index = OrdinalRegressionModel

setmetatable(OrdinalRegressionModel, GradientMethodBaseModel)

local defaultMaximumNumberOfIterations = 500

local defaultLearningRate = 0.1

local defaultBinaryFunction = "Logistic"

local defaultEpsilon = 1e-14

local function initializeThresholdVector(numberOfClasses)
	
	local unwrappedThresholdVector = {}
	
	for k = 1, (numberOfClasses - 1), 1 do unwrappedThresholdVector[k] = k end
	
	return {unwrappedThresholdVector}
	
end

local function enforceThresholdOrdering(thresholdVector, epsilon)
	
	local unwrappedThresholdVector = thresholdVector[1]
	
	local currentThresholdValue
	
	local previousThresholdValue
	
	local thresholdValueDifference
	
	for k = 2, #unwrappedThresholdVector, 1 do
		
		currentThresholdValue = unwrappedThresholdVector[k]
		
		previousThresholdValue = unwrappedThresholdVector[k - 1]
		
		if (unwrappedThresholdVector[k] <= previousThresholdValue) then
			
			thresholdValueDifference = math.abs(previousThresholdValue - currentThresholdValue)
			
			unwrappedThresholdVector[k] = previousThresholdValue + math.log(math.max(thresholdValueDifference, epsilon))
			
		end
		
	end
	
	return {unwrappedThresholdVector}
	
end

local function calculateProbabilityDensityFunctionValue(z)

	return (math.exp(-0.5 * math.pow(z, 2)) / math.sqrt(2 * math.pi))

end

local binaryFunctionList = {

	["Logistic"] = function (z) return (1/(1 + math.exp(-z))) end,
	
	["Logit"] = function (z) return (1/(1 + math.exp(-z))) end,
	
	["Probit"] = function(z) return ZTableFunction:getStandardNormalCumulativeDistributionFunction(math.clamp(z, -3.9, 3.9)) end,

	["LogLog"] = function(z) return math.exp(-math.exp(z)) end,

	["ComplementaryLogLog"] = function(z) return (1 - math.exp(-math.exp(z))) end,
	
	["HardSigmoid"] = function (z)

		local x = (z + 1) / 2

		if (x < 0) then return 0 elseif (x > 1) then return 1 else return x end

	end,

}

local binaryFunctionGradientList = {
	
	["Logistic"] = function (h, z) return (h * (1 - h)) end,
	
	["Logit"] = function (h, z) return (h * (1 - h)) end,
	
	["Probit"] = function (h, z) return calculateProbabilityDensityFunctionValue(z) end,

	["LogLog"] = function(h, z) return -math.exp(z) * math.exp(-math.exp(z)) end,

	["ComplementaryLogLog"] = function(h, z) return math.exp(z) * math.exp(-math.exp(z)) end,
	
	["HardSigmoid"] = function (h, z) return ((h <= 0 or h >= 1) and 0) or 0.5 end,
	
}

local function createClassesList(labelVector)

	local classesList = {}

	local value

	for i = 1, #labelVector, 1 do

		value = labelVector[i][1]

		if not table.find(classesList, value) then

			table.insert(classesList, value)

		end

	end

	return classesList

end

local function areNumbersOnlyInList(list)

	for i, value in ipairs(list) do

		if (typeof(value) ~= "number") then return false end

	end

	return true

end

local function checkIfAnyLabelVectorIsNotRecognized(labelVector, ClassesList)

	for i = 1, #labelVector, 1 do

		if (not table.find(ClassesList, labelVector[i][1])) then return true end

	end

	return false

end

function OrdinalRegressionModel:processLabelVector(labelVector)

	local ClassesList = self.ClassesList

	if (#ClassesList == 0) then

		ClassesList = createClassesList(labelVector)

		local areNumbersOnly = areNumbersOnlyInList(ClassesList)

		if (areNumbersOnly) then table.sort(ClassesList, function(a,b) return a < b end) end

		self.ClassesList = ClassesList

	else

		if checkIfAnyLabelVectorIsNotRecognized(labelVector, ClassesList) then error("A value does not exist in the model\'s classes list is present in the label vector.") end

	end
	
	return ClassesList

end

function OrdinalRegressionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.areGradientsSaved = true -- We need to coerce this because we're relying this to store our threshold vector gradient.
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	local NewOrdinalRegressionModel = GradientMethodBaseModel.new(parameterDictionary)

	setmetatable(NewOrdinalRegressionModel, OrdinalRegressionModel)
	
	NewOrdinalRegressionModel:setName("OrdinalRegression")
	
	local learningRate = parameterDictionary.learningRate or defaultLearningRate

	NewOrdinalRegressionModel.weightLearningRate = parameterDictionary.weightLearningRate or learningRate
	
	NewOrdinalRegressionModel.thresholdLearningRate = parameterDictionary.thresholdLearningRate or learningRate

	NewOrdinalRegressionModel.binaryFunction = parameterDictionary.binaryFunction or defaultBinaryFunction
	
	NewOrdinalRegressionModel.epsilon = parameterDictionary.epsilon or defaultEpsilon
	
	NewOrdinalRegressionModel.ClassesList = parameterDictionary.ClassesList or {}

	NewOrdinalRegressionModel.WeightOptimizer = parameterDictionary.WeightOptimizer
	
	NewOrdinalRegressionModel.ThresholdOptimizer = parameterDictionary.ThresholdOptimizer

	NewOrdinalRegressionModel.WeightRegularizer = parameterDictionary.WeightRegularizer
	
	NewOrdinalRegressionModel.ThresholdRegularizer = parameterDictionary.ThresholdRegularizer

	return NewOrdinalRegressionModel

end

function OrdinalRegressionModel:setWeightOptimizer(WeightOptimizer)

	self.WeightOptimizer = WeightOptimizer

end

function OrdinalRegressionModel:setThresholdOptimizer(ThresholdOptimizer)

	self.ThresholdOptimizer = ThresholdOptimizer

end

function OrdinalRegressionModel:setWeightRegularizer(WeightRegularizer)

	self.WeightRegularizer = WeightRegularizer

end

function OrdinalRegressionModel:setThresholdRegularizer(ThresholdRegularizer)

	self.ThresholdRegularizer = ThresholdRegularizer

end

function OrdinalRegressionModel:calculateCost(hypothesisMatrix, labelVector)

	local epsilon = self.epsilon

	local ClassesList = self.ClassesList
	
	local weightMatrix = self.ModelParameters[1]

	local totalCost = 0

	local labelValue

	local classIndex

	local probability

	for dataIndex, unwrappedHypothesisVector in ipairs(hypothesisMatrix) do

		labelValue = labelVector[dataIndex][1]

		classIndex = table.find(ClassesList, labelValue)

		probability = unwrappedHypothesisVector[classIndex] or 0

		-- Negative log-likelihood.

		totalCost = totalCost - math.log(math.max(probability, epsilon))

	end

	local WeightRegularizer = self.WeightRegularizer

	if (WeightRegularizer) then totalCost = totalCost + WeightRegularizer:calculateCost(weightMatrix) end

	local averageCost = totalCost / #labelVector

	return averageCost

end

function OrdinalRegressionModel:calculateHypothesisMatrix(featureMatrix, saveAllMatrices)

	local ClassesList = self.ClassesList

	local binaryFunctionToApply = binaryFunctionList[self.binaryFunction]

	local numberOfClasses = #ClassesList

	local numberOfClassesMinusOne = numberOfClasses - 1

	local ModelParameters = self.ModelParameters

	local weightMatrix = ModelParameters[1]

	local thresholdVector = ModelParameters[2]

	local unwrappedThresholdVector = thresholdVector[1]

	local zVector = AqwamTensorLibrary:dotProduct(featureMatrix, weightMatrix)

	local cumulativeProbabilityMatrix = {}

	local unwrappedCumulativeProbabilityVector

	local adjustedZValue

	for dataIndex, unwrappedZVector in ipairs(zVector) do

		unwrappedCumulativeProbabilityVector = {}

		for k = 1, numberOfClassesMinusOne, 1 do

			adjustedZValue = unwrappedThresholdVector[k] - unwrappedZVector[1]

			unwrappedCumulativeProbabilityVector[k] = binaryFunctionToApply(adjustedZValue)

		end

		cumulativeProbabilityMatrix[dataIndex] = unwrappedCumulativeProbabilityVector

	end

	-- Convert to category probabilities (K probabilities that sum to 1).

	local hypothesisMatrix = {}

	local unwrappedClassProbabilityVector = {}
	
	local probabilityDifference

	for dataIndex, cumulativeProbability in ipairs(cumulativeProbabilityMatrix) do

		unwrappedClassProbabilityVector = {}

		unwrappedClassProbabilityVector[1] = cumulativeProbability[1]  -- P(Y = 1) = P(Y ≤ 1)

		for k = 2, numberOfClassesMinusOne, 1 do
			
			probabilityDifference = cumulativeProbability[k] - cumulativeProbability[k-1] -- P(Y = k) = P(Y ≤ k) - P(Y ≤ k-1)

			unwrappedClassProbabilityVector[k] = math.clamp(probabilityDifference, 0, 1)

		end

		unwrappedClassProbabilityVector[numberOfClasses] = 1 - cumulativeProbability[numberOfClassesMinusOne]  -- P(Y = K) = 1 - P(Y ≤ K-1)

		hypothesisMatrix[dataIndex] = unwrappedClassProbabilityVector

	end

	if (saveAllMatrices) then

		self.featureMatrix = featureMatrix

		self.zVector = zVector

		self.hypothesisMatrix = hypothesisMatrix

		self.cumulativeProbabilityMatrix = cumulativeProbabilityMatrix

	end

	return hypothesisMatrix

end

function OrdinalRegressionModel:calculateLossFunctionDerivativeVector(lossGradientVector)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local featureMatrix = self.featureMatrix

	local zVector = self.zVector

	local cumulativeProbabilityMatrix = self.cumulativeProbabilityMatrix

	if (not featureMatrix) then error("Feature matrix not found.") end

	if (not zVector) then error("Z vector not found.") end

	if (not cumulativeProbabilityMatrix) then error("Cumulative probability matrix not found.") end

	local ClassesList = self.ClassesList

	local ModelParameters = self.ModelParameters

	local weightMatrix = ModelParameters[1]

	local thresholdVector = ModelParameters[2]

	local unwrappedThresholdVector = thresholdVector[1]

	local binaryFunctionGradientToApply = binaryFunctionGradientList[self.binaryFunction]

	local numberOfClasses = #ClassesList

	local numberOfClassesMinusOne = numberOfClasses - 1

	local lossFunctionDerivativeVector = AqwamTensorLibrary:createTensor({#featureMatrix[1], 1}, 0)

	local unwrappedThresholdGradientVector = {}

	for k = 1, numberOfClassesMinusOne do unwrappedThresholdGradientVector[k] = 0 end

	local unwrappedFeatureVector

	local unwrappedCumulativeProbabilityVector

	local trueLabelProbability

	local zValue

	local labelProbability

	local cumulativeProbability

	local probabilityDifference

	local adjustedZValue

	local binaryGradientValue

	for i = 1, #lossGradientVector do

		unwrappedFeatureVector = featureMatrix[i]

		unwrappedCumulativeProbabilityVector = cumulativeProbabilityMatrix[i]

		trueLabelProbability = lossGradientVector[i][1]  -- Actually the true category label!

		zValue = zVector[i][1]

		for k = 1, numberOfClassesMinusOne do

			labelProbability = ((trueLabelProbability <= k) and 1) or 0

			cumulativeProbability = unwrappedCumulativeProbabilityVector[k]

			probabilityDifference = cumulativeProbability - labelProbability

			adjustedZValue = unwrappedThresholdVector[k] - zValue

			binaryGradientValue = binaryFunctionGradientToApply(cumulativeProbability, adjustedZValue)

			for f, featureValue in ipairs(unwrappedFeatureVector) do

				lossFunctionDerivativeVector[f][1] = lossFunctionDerivativeVector[f][1] - (probabilityDifference * binaryGradientValue * featureValue)

			end

			unwrappedThresholdGradientVector[k] = unwrappedThresholdGradientVector[k] + (probabilityDifference * binaryGradientValue)

		end

	end
	
	local thresholdGradientVector = {unwrappedThresholdGradientVector}

	if (self.areGradientsSaved) then self.Gradients = {lossFunctionDerivativeVector, thresholdGradientVector} end

	return lossFunctionDerivativeVector

end

function OrdinalRegressionModel:gradientDescent(lossFunctionDerivativeVector, numberOfData)

	if (type(lossFunctionDerivativeVector) == "number") then lossFunctionDerivativeVector = {{lossFunctionDerivativeVector}} end

	local weightLearningRate = self.weightLearningRate

	local thresholdLearningRate = self.thresholdLearningRate

	local epsilon = self.epsilon

	local WeightRegularizer = self.WeightRegularizer

	local ThresholdRegularizer = self.ThresholdRegularizer

	local WeightOptimizer = self.WeightOptimizer

	local ThresholdOptimizer = self.ThresholdOptimizer

	local thresholdGradientVector = self.Gradients[2]

	local ModelParameters = self.ModelParameters

	local weightMatrix = ModelParameters[1]

	local thresholdVector = ModelParameters[2]

	if (WeightRegularizer) then

		local weightRegularizationDerivatives = WeightRegularizer:calculate(weightMatrix)

		lossFunctionDerivativeVector = AqwamTensorLibrary:add(lossFunctionDerivativeVector, weightRegularizationDerivatives)

	end

	if (ThresholdOptimizer) then

		local thresholdRegularizationDerivatives = ThresholdOptimizer:calculate(weightMatrix)

		thresholdGradientVector = AqwamTensorLibrary:add(thresholdGradientVector, thresholdRegularizationDerivatives)

	end

	lossFunctionDerivativeVector = AqwamTensorLibrary:divide(lossFunctionDerivativeVector, numberOfData)

	thresholdGradientVector = AqwamTensorLibrary:divide(thresholdGradientVector, numberOfData)

	if (WeightOptimizer) then

		lossFunctionDerivativeVector = WeightOptimizer:calculate(weightLearningRate, lossFunctionDerivativeVector, weightMatrix) 

	else

		lossFunctionDerivativeVector = AqwamTensorLibrary:multiply(weightLearningRate, lossFunctionDerivativeVector)

	end

	if (ThresholdOptimizer) then

		thresholdGradientVector = ThresholdOptimizer:calculate(thresholdLearningRate, thresholdGradientVector, thresholdVector) 

	else

		thresholdGradientVector = AqwamTensorLibrary:multiply(thresholdLearningRate, thresholdGradientVector)

	end

	local newWeightMatrix = AqwamTensorLibrary:subtract(weightMatrix, lossFunctionDerivativeVector)

	local newThresholdVector = AqwamTensorLibrary:subtract(thresholdVector, thresholdGradientVector)

	newThresholdVector = enforceThresholdOrdering(newThresholdVector, epsilon)

	self.ModelParameters = {newWeightMatrix, newThresholdVector}

end

function OrdinalRegressionModel:update(lossGradientVector, clearAllMatrices)

	if (type(lossGradientVector) == "number") then lossGradientVector = {{lossGradientVector}} end

	local numberOfData = #lossGradientVector

	local lossFunctionDerivativeVector = self:calculateLossFunctionDerivativeVector(lossGradientVector)

	self:gradientDescent(lossFunctionDerivativeVector, numberOfData)

	if (clearAllMatrices) then

		self.featureMatrix = nil

		self.zVector = nil

		self.hypothesisMatrix = nil

		self.cumulativeProbabilityMatrix = nil

	end

end

function OrdinalRegressionModel:train(featureMatrix, labelVector)

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end
	
	local ModelParameters = self.ModelParameters or {}
	
	local weightMatrix = ModelParameters[1]
	
	local thresholdVector = ModelParameters[2]
	
	local numberOfFeatures = #featureMatrix[1]
	
	local ClassesList = self:processLabelVector(labelVector)
	
	if (not weightMatrix) then

		ModelParameters[1] = self:initializeMatrixBasedOnMode({numberOfFeatures, 1}) 
		
	else
		
		if (numberOfFeatures ~= #weightMatrix) then error("The number of features are not the same as the weight matrix.") end

	end

	if (not thresholdVector) then ModelParameters[2] = initializeThresholdVector(#ClassesList) end
	
	self.ModelParameters = ModelParameters
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local WeightOptimizer = self.WeightOptimizer
	
	local ThresholdOptimizer = self.ThresholdOptimizer
	
	local costArray = {}

	local numberOfIterations = 0
	
	local cost
	
	repeat

		numberOfIterations = numberOfIterations + 1

		self:iterationWait()

		local hypothesisMatrix = self:calculateHypothesisMatrix(featureMatrix, true)

		cost = self:calculateCostWhenRequired(numberOfIterations, function()

			return self:calculateCost(hypothesisMatrix, labelVector)

		end)

		if (cost) then 

			table.insert(costArray, cost)

			self:printNumberOfIterationsAndCost(numberOfIterations, cost)

		end

		self:update(labelVector, true)

	until (numberOfIterations == maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)
	
	if (self.isOutputPrinted) then
		
		if (cost == math.huge) then warn("The model diverged.") end
		
		if (cost ~= cost) then warn("The model produced nan (not a number) values.") end
		
	end
	
	if (self.autoResetOptimizers) then
		
		if (WeightOptimizer) then WeightOptimizer:reset() end
		
		if (ThresholdOptimizer) then ThresholdOptimizer:reset() end
		
	end

	return costArray

end

function OrdinalRegressionModel:predict(featureMatrix, returnOriginalOutput)
	
	local ClassesList = self.ClassesList
	
	local ModelParameters = self.ModelParameters or {}
	
	local weightMatrix = ModelParameters[1]
	
	local thresholdVector = ModelParameters[2]

	if (not weightMatrix) then
		
		ModelParameters[1] = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1}) 
		
	end
	
	if (not thresholdVector) then

		ModelParameters[2] = initializeThresholdVector(#ClassesList)

	end
	
	self.ModelParameters = ModelParameters

	local probabilityMatrix = self:calculateHypothesisMatrix(featureMatrix, false)

	if (returnOriginalOutput) then return probabilityMatrix end
	
	local predictedLabelVector = {}
	
	local highestProbabilityVector = {}
	
	local highestProbability
	
	local classWithTheHighestIndex
	
	for i, unwrappedProbabilityVector in ipairs(probabilityMatrix) do
		
		highestProbability = -math.huge
		
		classWithTheHighestIndex = nil
		
		for class, probability in ipairs(unwrappedProbabilityVector) do
			
			if (probability > highestProbability) then
				
				highestProbability = probability
				
				classWithTheHighestIndex = class
				
			end
			
		end
		
		predictedLabelVector[i] = {ClassesList[classWithTheHighestIndex]}
		
		highestProbabilityVector[i] = {highestProbability}
		
	end

	return predictedLabelVector, highestProbabilityVector

end

return OrdinalRegressionModel
