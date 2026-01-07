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

local MatrixFactorizationBaseModel = require("Model_MatrixFactorizationBaseModel")

local FunkMatrixFactorizationModel = {}

FunkMatrixFactorizationModel.__index = FunkMatrixFactorizationModel

setmetatable(FunkMatrixFactorizationModel, MatrixFactorizationBaseModel)

local defaultMaximumNumberOfIterations = 500

local defaultLearningRate = 0.3

local defaultCostFunction = "MeanSquaredError"

local lossFunctionList = {

	["MeanSquaredError"] = function (h, y) return ((h - y)^2) end,

	["MeanAbsoluteError"] = function (h, y) return math.abs(h - y) end,

}

local lossFunctionGradientList = {

	["MeanSquaredError"] = function (h, y) return (2 * (h - y)) end,

	["MeanAbsoluteError"] = function (h, y) return math.sign(h - y) end,

}

function FunkMatrixFactorizationModel:calculateCost(hypothesisVector, labelVector)

	if (type(hypothesisVector) == "number") then hypothesisVector = {{hypothesisVector}} end

	local costVector = AqwamTensorLibrary:applyFunction(lossFunctionList[self.costFunction], hypothesisVector, labelVector)

	local totalCost = AqwamTensorLibrary:sum(costVector)
	
	local UserOptimizer = self.UserOptimizer
	
	local ItemOptimizer = self.ItemOptimizer
	
	local ModelParameters = self.ModelParameters or {}

	if (UserOptimizer) then totalCost = totalCost + UserOptimizer:calculateCost(ModelParameters[1]) end
	
	if (ItemOptimizer) then totalCost = totalCost + ItemOptimizer:calculateCost(ModelParameters[2]) end

	local averageCost = totalCost / (#labelVector * #labelVector[1])

	return averageCost

end

function FunkMatrixFactorizationModel:calculateHypothesisVector(userItemMatrix, saveUserItemMatrix)
	
	local latentFactorCount = self.latentFactorCount
	
	local ModelParameters = self.ModelParameters or {}
	
	local userLatentMatrix = ModelParameters[1] or self:initializeMatrixBasedOnMode({#userItemMatrix, latentFactorCount})

	local itemLatentMatrix = ModelParameters[2] or self:initializeMatrixBasedOnMode({latentFactorCount, #userItemMatrix[1]})

	local hypothesisVector = AqwamTensorLibrary:dotProduct(userLatentMatrix, itemLatentMatrix)
	
	self.ModelParameters = {userLatentMatrix, itemLatentMatrix}

	if (saveUserItemMatrix) then self.userItemMatrix = userItemMatrix end

	return hypothesisVector

end

function FunkMatrixFactorizationModel:calculateLossFunctionDerivativeVector(lossFunctionGradientMatrix)

	if (type(lossFunctionGradientMatrix) == "number") then lossFunctionGradientMatrix = {{lossFunctionGradientMatrix}} end
	
	local ModelParameters = self.ModelParameters

	local userLatentMatrix = ModelParameters[1]

	local itemLatentMatrix = ModelParameters[2]

	local userLossFunctionGradientMatrix = AqwamTensorLibrary:dotProduct(lossFunctionGradientMatrix, AqwamTensorLibrary:transpose(itemLatentMatrix))

	local itemLossFunctionGradientMatrix = AqwamTensorLibrary:dotProduct(AqwamTensorLibrary:transpose(userLatentMatrix), lossFunctionGradientMatrix)
	
	local lossFunctionDerivativeMatrixArray = {userLossFunctionGradientMatrix, itemLossFunctionGradientMatrix}

	if (self.areGradientsSaved) then self.lossFunctionDerivativeMatrixArray = lossFunctionDerivativeMatrixArray end

	return lossFunctionDerivativeMatrixArray

end

function FunkMatrixFactorizationModel:gradientDescent(lossFunctionDerivativeMatrixArray, numberOfData)
	
	local UserRegularizer = self.UserRegularizer

	local ItemRegularizer = self.ItemRegularizer

	local UserOptimizer = self.UserOptimizer

	local ItemOptimizer = self.ItemOptimizer

	local userLearningRate = self.userLearningRate

	local itemLearningRate = self.itemLearningRate
	
	local ModelParameters = self.ModelParameters
	
	local userLatentMatrix = ModelParameters[1]

	local itemLatentMatrix = ModelParameters[2]
	
	local userLatentLossFunctionDerivativeMatrix = lossFunctionDerivativeMatrixArray[1]
	
	local itemLatentLossFunctionDerivativeMatrix = lossFunctionDerivativeMatrixArray[2]
	
	if (UserRegularizer) then

		local userRegularizationDerivatives = UserRegularizer:calculate(userLatentMatrix)

		userLatentLossFunctionDerivativeMatrix = AqwamTensorLibrary:add(userLatentLossFunctionDerivativeMatrix, userRegularizationDerivatives)

	end
	
	if (ItemRegularizer) then

		local itemRegularizationDerivatives = ItemRegularizer:calculate(itemLatentMatrix)

		itemLatentLossFunctionDerivativeMatrix = AqwamTensorLibrary:add(itemLatentLossFunctionDerivativeMatrix, itemRegularizationDerivatives)

	end

	if (UserOptimizer) then 

		userLatentLossFunctionDerivativeMatrix = UserOptimizer:calculate(userLearningRate, userLatentLossFunctionDerivativeMatrix, userLatentMatrix) 

	else

		userLatentLossFunctionDerivativeMatrix = AqwamTensorLibrary:multiply(userLearningRate, userLatentLossFunctionDerivativeMatrix)

	end
	
	if (ItemOptimizer) then 

		itemLatentLossFunctionDerivativeMatrix = ItemOptimizer:calculate(itemLearningRate, itemLatentLossFunctionDerivativeMatrix, itemLatentMatrix) 

	else

		itemLatentLossFunctionDerivativeMatrix = AqwamTensorLibrary:multiply(itemLearningRate, itemLatentLossFunctionDerivativeMatrix)

	end
	
	userLatentLossFunctionDerivativeMatrix = AqwamTensorLibrary:divide(userLatentLossFunctionDerivativeMatrix, numberOfData)
	
	itemLatentLossFunctionDerivativeMatrix = AqwamTensorLibrary:divide(itemLatentLossFunctionDerivativeMatrix, numberOfData)
	
	userLatentMatrix = AqwamTensorLibrary:subtract(userLatentMatrix, userLatentLossFunctionDerivativeMatrix)
	
	itemLatentMatrix = AqwamTensorLibrary:subtract(itemLatentMatrix, itemLatentLossFunctionDerivativeMatrix)

	self.ModelParameters = {userLatentMatrix, itemLatentMatrix}

end

function FunkMatrixFactorizationModel:update(lossGradientMatrix, clearAllMatrices)

	if (type(lossGradientMatrix) == "number") then lossGradientMatrix = {{lossGradientMatrix}} end

	local lossFunctionDerivativeMatrixArray = self:calculateLossFunctionDerivativeVector(lossGradientMatrix)
	
	local numberOfData = #lossGradientMatrix * #lossGradientMatrix[1]

	self:gradientDescent(lossFunctionDerivativeMatrixArray, numberOfData)

	if (clearAllMatrices) then 

		self.userItemMatrix = nil 

		self.lossFunctionDerivativeMatrixArray = nil

	end

end

function FunkMatrixFactorizationModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	local NewFunkMatrixFactorizationModel = MatrixFactorizationBaseModel.new(parameterDictionary)

	setmetatable(NewFunkMatrixFactorizationModel, FunkMatrixFactorizationModel)
	
	NewFunkMatrixFactorizationModel:setName("FunkMatrixFactorization")
	
	local learningRate = parameterDictionary.learningRate or defaultLearningRate

	NewFunkMatrixFactorizationModel.costFunction = parameterDictionary.costFunction or defaultCostFunction
	
	NewFunkMatrixFactorizationModel.userLearningRate = parameterDictionary.userLearningRate or learningRate
	
	NewFunkMatrixFactorizationModel.itemLearningRate = parameterDictionary.itemLearningRate or learningRate

	NewFunkMatrixFactorizationModel.UserOptimizer = parameterDictionary.UserOptimizer
	
	NewFunkMatrixFactorizationModel.ItemOptimizer = parameterDictionary.ItemOptimizer

	NewFunkMatrixFactorizationModel.UserRegularizer = parameterDictionary.UserRegularizer
	
	NewFunkMatrixFactorizationModel.ItemRegularizer = parameterDictionary.ItemRegularizer

	return NewFunkMatrixFactorizationModel

end

function FunkMatrixFactorizationModel:setUserOptimizer(UserOptimizer)

	self.UserOptimizer = UserOptimizer

end

function FunkMatrixFactorizationModel:setItemOptimizer(ItemOptimizer)

	self.ItemOptimizer = ItemOptimizer

end

function FunkMatrixFactorizationModel:setUserRegularizer(UserRegularizer)

	self.UserRegularizer = UserRegularizer

end

function FunkMatrixFactorizationModel:setItemRegularizer(ItemRegularizer)

	self.ItemRegularizer = ItemRegularizer

end

function FunkMatrixFactorizationModel:train(userItemDictionaryDictionary)
	
	local lossFunctionGradientFunctionToApply = lossFunctionGradientList[self.costFunction]

	if (not lossFunctionGradientFunctionToApply) then error("Invalid cost function.") end
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local latentFactorCount = self.latentFactorCount

	local UserOptimizer = self.UserOptimizer
	
	local ItemOptimizer = self.ItemOptimizer
	
	local userItemMatrix, numberOfUserIDsAdded, numberOfItemIDsAdded = self:processUserItemDictionaryDictionary(userItemDictionaryDictionary)
	
	local ModelParameters = self.ModelParameters or {}

	local userLatentMatrix = ModelParameters[1]

	local itemLatentMatrix = ModelParameters[2]
	
	if (numberOfUserIDsAdded >= 1) and (userLatentMatrix) then
		
		local userLatentSubMatrix = self:initializeMatrixBasedOnMode({numberOfUserIDsAdded, latentFactorCount})
		
		userLatentMatrix = AqwamTensorLibrary:concatenate(userLatentMatrix, userLatentSubMatrix, 1)
		
		ModelParameters[1] = userLatentMatrix
		
	end
	
	if (numberOfItemIDsAdded >= 1) and (itemLatentMatrix) then

		local itemLatentSubMatrix = self:initializeMatrixBasedOnMode({latentFactorCount, numberOfItemIDsAdded})

		itemLatentMatrix = AqwamTensorLibrary:concatenate(itemLatentMatrix, itemLatentSubMatrix, 2)
		
		ModelParameters[2] = itemLatentMatrix

	end

	local costArray = {}

	local numberOfIterations = 0
	
	local cost

	repeat

		numberOfIterations = numberOfIterations + 1

		self:iterationWait()

		local hypothesisVector = self:calculateHypothesisVector(userItemMatrix, true)

		cost = self:calculateCostWhenRequired(numberOfIterations, function()

			return self:calculateCost(hypothesisVector, userItemMatrix)

		end)

		if (cost) then 

			table.insert(costArray, cost)

			self:printNumberOfIterationsAndCost(numberOfIterations, cost)

		end

		local lossGradientVector = AqwamTensorLibrary:applyFunction(lossFunctionGradientFunctionToApply, hypothesisVector, userItemMatrix)

		self:update(lossGradientVector, true)

	until (numberOfIterations == maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)

	if (self.isOutputPrinted) then

		if (cost == math.huge) then warn("The model diverged.") end

		if (cost ~= cost) then warn("The model produced nan (not a number) values.") end

	end
	
	if (self.autoResetOptimizers) then
		
		if (UserOptimizer) then UserOptimizer:reset() end
		
		if (ItemOptimizer) then ItemOptimizer:reset() end
		
	end
	
	return costArray

end

function FunkMatrixFactorizationModel:predict(userIDVector, returnOriginalOutput)
	
	local storedUserIDArray = self.userIDArray

	local storedItemIDArray = self.itemIDArray
	
	local ModelParameters = self.ModelParameters or {}

	local userLatentMatrix = ModelParameters[1]

	local itemLatentMatrix = ModelParameters[2]
	
	if (userIDVector) then
		
		local userLatentSubMatrix = {}
		
		for i, unwrappedUserIDVector in ipairs(userIDVector) do
			
			local targetIndex = table.find(storedUserIDArray, unwrappedUserIDVector[1])
			
			if (targetIndex) then userLatentSubMatrix[i] = userLatentMatrix[targetIndex] end
			
		end
		
		userLatentMatrix = userLatentSubMatrix
		
	end
	
	local predictedMatrix = AqwamTensorLibrary:dotProduct(userLatentMatrix, itemLatentMatrix)

	if (returnOriginalOutput) then return predictedMatrix end
	
	return self:fetchHighestValueVector(predictedMatrix)

end

return FunkMatrixFactorizationModel
