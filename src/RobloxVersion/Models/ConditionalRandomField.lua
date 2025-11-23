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

local GradientMethodBaseModel = require(script.Parent.GradientMethodBaseModel)

local ConditionalRandomFieldModel = {}

ConditionalRandomFieldModel.__index = ConditionalRandomFieldModel

setmetatable(ConditionalRandomFieldModel, GradientMethodBaseModel)

local defaultMaximumNumberOfIterations = 500

local defaultLearningRate = 0.1

local defaultAddBias = true

local defaultAddStabilization = true

function ConditionalRandomFieldModel:calculateCost(predictedCurrentStateMatrix, currentStateMatrix)
	
	local logPredictedCurrentStateMatrix = AqwamTensorLibrary:applyFunction(math.log, predictedCurrentStateMatrix)
	
	local costMatrix = AqwamTensorLibrary:multiply(currentStateMatrix, logPredictedCurrentStateMatrix)
	
	local totalCost = -AqwamTensorLibrary:sum(costMatrix)
	
	local Regularizer = self.Regularizer

	if (Regularizer) then totalCost = totalCost + Regularizer:calculateCost(self.ModelParameters) end

	local averageCost = totalCost / #currentStateMatrix

	return averageCost

end

function ConditionalRandomFieldModel:calculateNextStateMatrix(stateMatrix, saveStateMatrix)

	local zMatrix = AqwamTensorLibrary:dotProduct(stateMatrix, self.ModelParameters)
	
	if (self.addStabilization) then

		local maximumZVector = AqwamTensorLibrary:findMaximumValue(zMatrix, 2)

		zMatrix = AqwamTensorLibrary:subtract(zMatrix, maximumZVector)

	end
	
	local exponentMatrix = AqwamTensorLibrary:applyFunction(math.exp, zMatrix)
	
	local sumExponentVector = AqwamTensorLibrary:sum(exponentMatrix, 2)
	
	local nextStateMatrix = AqwamTensorLibrary:divide(exponentMatrix, sumExponentVector)

	if (saveStateMatrix) then 

		self.stateMatrix = stateMatrix

	end

	return nextStateMatrix

end

function ConditionalRandomFieldModel:calculateLossFunctionDerivativeMatrix(lossGradientMatrix)

	if (type(lossGradientMatrix) == "number") then lossGradientMatrix = {{lossGradientMatrix}} end

	local stateMatrix = self.stateMatrix

	if (not stateMatrix) then error("State matrix not found.") end

	local lossFunctionDerivativeMatrix = AqwamTensorLibrary:dotProduct(AqwamTensorLibrary:transpose(stateMatrix), lossGradientMatrix)

	if (self.areGradientsSaved) then self.Gradients = lossFunctionDerivativeMatrix end

	return lossFunctionDerivativeMatrix

end

function ConditionalRandomFieldModel:gradientDescent(lossFunctionDerivativeMatrix, numberOfData)

	if (type(lossFunctionDerivativeMatrix) == "number") then lossFunctionDerivativeMatrix = {{lossFunctionDerivativeMatrix}} end
	
	local ModelParameters = self.ModelParameters

	local Regularizer = self.Regularizer

	local Optimizer = self.Optimizer

	local learningRate = self.learningRate
	
	if (Regularizer) then

		local regularizationDerivatives = Regularizer:calculate(ModelParameters)

		lossFunctionDerivativeMatrix = AqwamTensorLibrary:add(lossFunctionDerivativeMatrix, regularizationDerivatives)

	end

	lossFunctionDerivativeMatrix = AqwamTensorLibrary:divide(lossFunctionDerivativeMatrix, numberOfData)

	if (Optimizer) then

		lossFunctionDerivativeMatrix = Optimizer:calculate(learningRate, lossFunctionDerivativeMatrix, ModelParameters) 

	else

		lossFunctionDerivativeMatrix = AqwamTensorLibrary:multiply(learningRate, lossFunctionDerivativeMatrix)

	end

	self.ModelParameters = AqwamTensorLibrary:subtract(ModelParameters, lossFunctionDerivativeMatrix)

end

function ConditionalRandomFieldModel:update(lossGradientMatrix, clearFeatureMatrix)

	if (type(lossGradientMatrix) == "number") then lossGradientMatrix = {{lossGradientMatrix}} end

	local numberOfData = #lossGradientMatrix

	local lossFunctionDerivativeMatrix = self:calculateLossFunctionDerivativeMatrix(lossGradientMatrix)

	self:gradientDescent(lossFunctionDerivativeMatrix, numberOfData)
	
	if (clearFeatureMatrix) then self.featureMatrix = nil end

end

function ConditionalRandomFieldModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	local NewConditionalRandomFieldModel = GradientMethodBaseModel.new(parameterDictionary)

	setmetatable(NewConditionalRandomFieldModel, ConditionalRandomFieldModel)
	
	NewConditionalRandomFieldModel:setName("ConditionalRandomField")

	NewConditionalRandomFieldModel.learningRate = parameterDictionary.learningRate or defaultLearningRate
	
	NewConditionalRandomFieldModel.addBias = NewConditionalRandomFieldModel:getValueOrDefaultValue(parameterDictionary.addBias, defaultAddBias)
	
	NewConditionalRandomFieldModel.addStabilization = NewConditionalRandomFieldModel:getValueOrDefaultValue(parameterDictionary.addStabilization, defaultAddStabilization)

	NewConditionalRandomFieldModel.Optimizer = parameterDictionary.Optimizer

	NewConditionalRandomFieldModel.Regularizer = parameterDictionary.Regularizer

	return NewConditionalRandomFieldModel

end

function ConditionalRandomFieldModel:setOptimizer(Optimizer)

	self.Optimizer = Optimizer

end

function ConditionalRandomFieldModel:setRegularizer(Regularizer)

	self.Regularizer = Regularizer

end

function ConditionalRandomFieldModel:train(previousStateMatrix, currentStateMatrix)
	
	local numberOfData = #previousStateMatrix

	if (numberOfData ~= #currentStateMatrix) then error("The number of data in the previous state vector is not equal to the number of data in the current state vector.") end
	
	local addBias = self.addBias
	
	local hasBiasValue = (addBias and 1) or 0
	
	local ModelParameters = self.ModelParameters

	local numberOfPreviousStateColumns = #previousStateMatrix[1]

	local numberOfCurrentStateColumns = #currentStateMatrix[1]

	local numberOfStates

	if (not ModelParameters) then

		numberOfStates = numberOfCurrentStateColumns

		self.ModelParameters = self:initializeMatrixBasedOnMode({numberOfStates + hasBiasValue, numberOfStates})
		
	else
		
		numberOfStates = #ModelParameters

	end

	if (numberOfPreviousStateColumns ~= numberOfStates) then
		
		if (addBias) and ((numberOfPreviousStateColumns - numberOfStates) == 1) then
			
			error("The number of previous state columns is not equal to the number of states and bias.") 
			
		else
			
			error("The number of previous state columns is not equal to the number of states.") 
			
		end
		
	end

	if (numberOfCurrentStateColumns ~= numberOfStates) then error("The number of current state columns is not equal to the number of states.") end
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local Optimizer = self.Optimizer
	
	local costArray = {}

	local numberOfIterations = 0
	
	local cost
	
	repeat

		numberOfIterations = numberOfIterations + 1

		self:iterationWait()

		local predictedCurrentStateMatrix = self:calculateNextStateMatrix(previousStateMatrix, true)

		cost = self:calculateCostWhenRequired(numberOfIterations, function()

			return self:calculateCost(predictedCurrentStateMatrix, currentStateMatrix)

		end)

		if (cost) then 

			table.insert(costArray, cost)

			self:printNumberOfIterationsAndCost(numberOfIterations, cost)

		end

		local lossGradientMatrix = AqwamTensorLibrary:subtract(predictedCurrentStateMatrix, currentStateMatrix)

		self:update(lossGradientMatrix, true)

	until (numberOfIterations == maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)
	
	if (self.isOutputPrinted) then
		
		if (cost == math.huge) then warn("The model diverged.") end
		
		if (cost ~= cost) then warn("The model produced nan (not a number) values.") end
		
	end

	if (Optimizer) and (self.autoResetOptimizers) then Optimizer:reset() end

	return costArray

end

function ConditionalRandomFieldModel:predict(stateMatrix)
	
	if (not self.ModelParameters) then 
		
		local numberOfStates = #stateMatrix[1]
		
		local hasBiasValue = (self.addBias and 1) or 0
		
		self.ModelParameters = self:initializeMatrixBasedOnMode({numberOfStates + hasBiasValue, numberOfStates}) 
		
	end

	local nextStateMatrix = self:calculateNextStateMatrix(stateMatrix, false)

	return nextStateMatrix

end

return ConditionalRandomFieldModel
