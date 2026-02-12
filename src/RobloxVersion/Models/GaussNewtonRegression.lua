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

local GaussNewtonRegressionModel = {}

GaussNewtonRegressionModel.__index = GaussNewtonRegressionModel

setmetatable(GaussNewtonRegressionModel, IterativeMethodBaseModel)

local defaultMaximumNumberOfIterations = 500

local defaultLearningRate = 1

local defaultModelParametersInitializationMode = "Zero"

function GaussNewtonRegressionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	parameterDictionary.modelParametersInitializationMode = parameterDictionary.modelParametersInitializationMode or defaultModelParametersInitializationMode

	local NewGaussNewtonRegressionModel = IterativeMethodBaseModel.new(parameterDictionary)

	setmetatable(NewGaussNewtonRegressionModel, GaussNewtonRegressionModel)

	NewGaussNewtonRegressionModel:setName("GaussNewtonRegression")
	
	NewGaussNewtonRegressionModel.learningRate = parameterDictionary.learningRate or defaultLearningRate

	return NewGaussNewtonRegressionModel

end

function GaussNewtonRegressionModel:train(featureMatrix, labelVector)

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end
	
	local numberOfFeatures = #featureMatrix[1]
	
	local betaVector = self.ModelParameters

	if (betaVector) then

		if (numberOfFeatures ~= #betaVector) then error("The number of features are not the same as the model parameters.") end

	else

		betaVector = self:initializeMatrixBasedOnMode({numberOfFeatures, 1})

	end
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local learningRate = self.learningRate
	
	local costArray = {}
	
	local numberOfIterations = 0
	
	local transposedFeatureMatrix
	
	local dotProductFeatureMatrix
	
	local inverseDotProductMatrix
	
	local responseVector
	
	local errorVector
	
	local squaredErrorVector
	
	local betaChangeVector
	
	local cost
	
	repeat

		numberOfIterations = numberOfIterations + 1

		self:iterationWait()
		
		transposedFeatureMatrix = AqwamTensorLibrary:transpose(featureMatrix)
		
		dotProductFeatureMatrix = AqwamTensorLibrary:dotProduct(transposedFeatureMatrix, featureMatrix)
		
		inverseDotProductMatrix = AqwamTensorLibrary:inverse(dotProductFeatureMatrix)
		
		responseVector = AqwamTensorLibrary:dotProduct(featureMatrix, betaVector)
		
		errorVector = AqwamTensorLibrary:subtract(labelVector, responseVector)
		
		betaChangeVector = AqwamTensorLibrary:dotProduct(inverseDotProductMatrix, transposedFeatureMatrix, errorVector)
		
		betaChangeVector = AqwamTensorLibrary:multiply(learningRate, betaChangeVector)
		
		betaVector = AqwamTensorLibrary:add(betaVector, betaChangeVector)
		
		cost = self:calculateCostWhenRequired(numberOfIterations, function()
			
			squaredErrorVector = AqwamTensorLibrary:power(errorVector, 2)

			return AqwamTensorLibrary:sum(squaredErrorVector)

		end)

		if (cost) then 

			table.insert(costArray, cost)

			self:printNumberOfIterationsAndCost(numberOfIterations, cost)

		end
		
	until (numberOfIterations == maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)
	
	if (self.isOutputPrinted) then

		if (cost == math.huge) then warn("The model diverged.") end

		if (cost ~= cost) then warn("The model produced nan (not a number) values.") end

	end

	self.ModelParameters = betaVector
	
	return costArray

end

function GaussNewtonRegressionModel:predict(featureMatrix)

	local betaVector = self.ModelParameters

	if (not betaVector) then

		betaVector = self:initializeMatrixBasedOnMode({#featureMatrix[1], 1})

		self.ModelParameters = betaVector

	end

	return AqwamTensorLibrary:dotProduct(featureMatrix, betaVector)

end

return GaussNewtonRegressionModel
