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

local LeastTrimmedSquaresRegressionModel = {}

LeastTrimmedSquaresRegressionModel.__index = LeastTrimmedSquaresRegressionModel

setmetatable(LeastTrimmedSquaresRegressionModel, IterativeMethodBaseModel)

local defaultMaximumNumberOfIterations = 300

local defaultNumberOfPoints = math.huge

local defaultLambda = 0

local function calculatePseudoInverseMatrix(matrix, lambdaIdentityMatrix)
	
	local transposedMatrix = AqwamTensorLibrary:transpose(matrix)
	
	local pseudoInverseMatrix = AqwamTensorLibrary:dotProduct(transposedMatrix, matrix)
	
	if (lambdaIdentityMatrix) then pseudoInverseMatrix = AqwamTensorLibrary:add(pseudoInverseMatrix, lambdaIdentityMatrix) end
	
	pseudoInverseMatrix = AqwamTensorLibrary:inverse(pseudoInverseMatrix)
	
	if (not pseudoInverseMatrix) then return nil end
	
	pseudoInverseMatrix = AqwamTensorLibrary:dotProduct(pseudoInverseMatrix, transposedMatrix)
	
	return pseudoInverseMatrix
	
end

function calculateCost(featureMatrix, labelVector, ModelParameters, numberOfPoints)

	local responseVector = AqwamTensorLibrary:dotProduct(featureMatrix, ModelParameters)
	
	local residualVector = AqwamTensorLibrary:subtract(labelVector, responseVector)
	
	local squaredResidualVector = AqwamTensorLibrary:power(residualVector, 2)
	
	local transposedSquaredResidualVector = AqwamTensorLibrary:transpose(squaredResidualVector)
	
	local residualValueArray = transposedSquaredResidualVector[1]
	
	table.sort(residualValueArray)

	local cost = 0
	
	for i = 1, numberOfPoints, 1 do cost = cost + residualValueArray[i] end

	return cost
end

function LeastTrimmedSquaresRegressionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations
	
	local NewLeastTrimmedSquaresRegressionModel = IterativeMethodBaseModel.new(parameterDictionary)
	
	setmetatable(NewLeastTrimmedSquaresRegressionModel, LeastTrimmedSquaresRegressionModel)
	
	NewLeastTrimmedSquaresRegressionModel:setName("LeastTrimmedSquaresRegression")
	
	NewLeastTrimmedSquaresRegressionModel.numberOfPoints = parameterDictionary.numberOfPoints or defaultNumberOfPoints
	
	NewLeastTrimmedSquaresRegressionModel.lambda = parameterDictionary.lambda or defaultLambda
	
	return NewLeastTrimmedSquaresRegressionModel
	
end

function LeastTrimmedSquaresRegressionModel:train(featureMatrix, labelVector)
	
	local numberOfData = #featureMatrix
	
	if (numberOfData ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end
	
	local numberOfFeatures = #featureMatrix[1]
	
	local ModelParameters = self.ModelParameters
	
	if (ModelParameters) then

		if (numberOfFeatures ~= #ModelParameters) then error("The number of features are not the same as the model parameters.") end

	else

		ModelParameters = self:initializeMatrixBasedOnMode({numberOfFeatures, 1})

	end
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local numberOfPoints = self.numberOfPoints
	
	local lambda = self.lambda
	
	if (numberOfPoints == math.huge) then numberOfPoints = math.floor((numberOfData / 2) + ((numberOfFeatures + 1) / 2)) end
	
	local subLambdaIdentityMatrix
	
	if (lambda ~= 0) then subLambdaIdentityMatrix = AqwamTensorLibrary:createIdentityTensor({numberOfFeatures, numberOfFeatures}, lambda) end
	
	local costArray = {}

	local numberOfIterations = 0
	
	local cost
	
	local residualIndexDictionary
	
	local residualIndexDictionaryArray 
	
	local subFeatureMatrix
	
	local subLabelVector
	
	local pseudoInverseSubFeatureMatrix
	
	local responseVector
	
	local residualVector
	
	local squaredResidualVector
	
	local index
	
	repeat
		
		self:iterationWait()
		
		numberOfIterations = numberOfIterations + 1
		
		responseVector = AqwamTensorLibrary:dotProduct(featureMatrix, ModelParameters)
		
		residualVector = AqwamTensorLibrary:subtract(labelVector, responseVector)

		squaredResidualVector = AqwamTensorLibrary:power(residualVector, 2)

		residualIndexDictionaryArray = {}
		
		for i, unwrappedSquaredResidualVector in ipairs(squaredResidualVector) do 
			
			table.insert(residualIndexDictionaryArray, {index = i, value = unwrappedSquaredResidualVector[1]}) 
			
		end

		table.sort(residualIndexDictionaryArray, function(a, b) return a.value < b.value end)
		
		subFeatureMatrix = {}
		
		subLabelVector = {}

		for i = 1, numberOfPoints, 1 do
			
			residualIndexDictionary = residualIndexDictionaryArray[i]
			
			index = residualIndexDictionary.index
			
			subFeatureMatrix[i] = featureMatrix[index]
			
			subLabelVector[i] = labelVector[index]
			
		end

		pseudoInverseSubFeatureMatrix = calculatePseudoInverseMatrix(subFeatureMatrix, subLambdaIdentityMatrix)

		if (pseudoInverseSubFeatureMatrix) then
			
			ModelParameters = AqwamTensorLibrary:dotProduct(pseudoInverseSubFeatureMatrix, subLabelVector)
			
			cost = self:calculateCostWhenRequired(numberOfIterations, function()

				return calculateCost(featureMatrix, labelVector, ModelParameters, numberOfPoints)

			end)
			
			if (cost) then 

				table.insert(costArray, cost)

				self:printNumberOfIterationsAndCost(numberOfIterations, cost)

			end
			
		end
		
	until (numberOfIterations >= maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost) or self:checkIfNan(cost)
	
	if (self.isOutputPrinted) then

		if (cost == math.huge) then warn("The model diverged.") end

		if (cost ~= cost) then warn("The model produced nan (not a number) values.") end

	end
	
	if (self.autoResetConvergenceCheck) then self:resetConvergenceCheck() end
	
	self.ModelParameters = ModelParameters
	
	return costArray
	
end

function LeastTrimmedSquaresRegressionModel:predict(featureMatrix)
	
	local ModelParameters = self.ModelParameters

	if (not ModelParameters) then

		ModelParameters = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})

		self.ModelParameters = ModelParameters

	end

	return AqwamTensorLibrary:dotProduct(featureMatrix, ModelParameters)
	
end

return LeastTrimmedSquaresRegressionModel
