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

local MatrixFactorizationBaseModel = require(script.Parent.MatrixFactorizationBaseModel)

local SimonFunkMatrixFactorizationModel = {}

SimonFunkMatrixFactorizationModel.__index = SimonFunkMatrixFactorizationModel

setmetatable(SimonFunkMatrixFactorizationModel, MatrixFactorizationBaseModel)

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

function SimonFunkMatrixFactorizationModel:calculateCost(hypothesisMatrix, labelMatrix, userItemMaskMatrix)

	if (type(hypothesisMatrix) == "number") then hypothesisMatrix = {{hypothesisMatrix}} end

	local costMatrix = AqwamTensorLibrary:applyFunction(lossFunctionList[self.costFunction], hypothesisMatrix, labelMatrix)
	
	costMatrix = AqwamTensorLibrary:multiply(costMatrix, userItemMaskMatrix)

	local totalCost = AqwamTensorLibrary:sum(costMatrix)
	
	local UserOptimizer = self.UserOptimizer
	
	local ItemOptimizer = self.ItemOptimizer
	
	local ModelParameters = self.ModelParameters or {}

	if (UserOptimizer) then totalCost = totalCost + UserOptimizer:calculateCost(ModelParameters[1]) end
	
	if (ItemOptimizer) then totalCost = totalCost + ItemOptimizer:calculateCost(ModelParameters[2]) end

	local averageCost = totalCost / (#labelMatrix * #labelMatrix[1])

	return averageCost

end

function SimonFunkMatrixFactorizationModel:calculateHypothesisMatrix(userItemMatrix, saveUserItemMatrix)
	
	local latentFactorCount = self.latentFactorCount
	
	local ModelParameters = self.ModelParameters or {}
	
	local userLatentMatrix = ModelParameters[1] or self:initializeMatrixBasedOnMode({#userItemMatrix, latentFactorCount})

	local itemLatentMatrix = ModelParameters[2] or self:initializeMatrixBasedOnMode({latentFactorCount, #userItemMatrix[1]})

	local hypothesisMatrix = AqwamTensorLibrary:dotProduct(userLatentMatrix, itemLatentMatrix)
	
	self.ModelParameters = {userLatentMatrix, itemLatentMatrix}

	if (saveUserItemMatrix) then self.userItemMatrix = userItemMatrix end

	return hypothesisMatrix

end

function SimonFunkMatrixFactorizationModel:calculateLossFunctionGradientVector(lossFunctionGradientMatrix)

	if (type(lossFunctionGradientMatrix) == "number") then lossFunctionGradientMatrix = {{lossFunctionGradientMatrix}} end
	
	local ModelParameters = self.ModelParameters

	local userLatentMatrix = ModelParameters[1]

	local itemLatentMatrix = ModelParameters[2]

	local userLossFunctionGradientMatrix = AqwamTensorLibrary:dotProduct(lossFunctionGradientMatrix, AqwamTensorLibrary:transpose(itemLatentMatrix))

	local itemLossFunctionGradientMatrix = AqwamTensorLibrary:dotProduct(AqwamTensorLibrary:transpose(userLatentMatrix), lossFunctionGradientMatrix)
	
	local lossFunctionGradientMatrixArray = {userLossFunctionGradientMatrix, itemLossFunctionGradientMatrix}

	if (self.areGradientsSaved) then self.lossFunctionGradientMatrixArray = lossFunctionGradientMatrixArray end

	return lossFunctionGradientMatrixArray

end

function SimonFunkMatrixFactorizationModel:gradientDescent(lossFunctionGradientMatrixArray, numberOfData)
	
	local UserRegularizer = self.UserRegularizer

	local ItemRegularizer = self.ItemRegularizer

	local UserOptimizer = self.UserOptimizer

	local ItemOptimizer = self.ItemOptimizer

	local userLearningRate = self.userLearningRate

	local itemLearningRate = self.itemLearningRate
	
	local ModelParameters = self.ModelParameters
	
	local userLatentMatrix = ModelParameters[1]

	local itemLatentMatrix = ModelParameters[2]
	
	local userLatentLossFunctionGradientMatrix = lossFunctionGradientMatrixArray[1]
	
	local itemLatentLossFunctionGradientMatrix = lossFunctionGradientMatrixArray[2]
	
	if (UserRegularizer) then

		local userRegularizationGradients = UserRegularizer:calculate(userLatentMatrix)

		userLatentLossFunctionGradientMatrix = AqwamTensorLibrary:add(userLatentLossFunctionGradientMatrix, userRegularizationGradients)

	end
	
	if (ItemRegularizer) then

		local itemRegularizationGradients = ItemRegularizer:calculate(itemLatentMatrix)

		itemLatentLossFunctionGradientMatrix = AqwamTensorLibrary:add(itemLatentLossFunctionGradientMatrix, itemRegularizationGradients)

	end

	if (UserOptimizer) then 

		userLatentLossFunctionGradientMatrix = UserOptimizer:calculate(userLearningRate, userLatentLossFunctionGradientMatrix, userLatentMatrix) 

	else

		userLatentLossFunctionGradientMatrix = AqwamTensorLibrary:multiply(userLearningRate, userLatentLossFunctionGradientMatrix)

	end
	
	if (ItemOptimizer) then 

		itemLatentLossFunctionGradientMatrix = ItemOptimizer:calculate(itemLearningRate, itemLatentLossFunctionGradientMatrix, itemLatentMatrix) 

	else

		itemLatentLossFunctionGradientMatrix = AqwamTensorLibrary:multiply(itemLearningRate, itemLatentLossFunctionGradientMatrix)

	end
	
	userLatentLossFunctionGradientMatrix = AqwamTensorLibrary:divide(userLatentLossFunctionGradientMatrix, numberOfData)
	
	itemLatentLossFunctionGradientMatrix = AqwamTensorLibrary:divide(itemLatentLossFunctionGradientMatrix, numberOfData)
	
	userLatentMatrix = AqwamTensorLibrary:subtract(userLatentMatrix, userLatentLossFunctionGradientMatrix)
	
	itemLatentMatrix = AqwamTensorLibrary:subtract(itemLatentMatrix, itemLatentLossFunctionGradientMatrix)

	self.ModelParameters = {userLatentMatrix, itemLatentMatrix}

end

function SimonFunkMatrixFactorizationModel:update(lossGradientMatrix, clearAllMatrices)

	if (type(lossGradientMatrix) == "number") then lossGradientMatrix = {{lossGradientMatrix}} end

	local lossFunctionGradientMatrixArray = self:calculateLossFunctionGradientVector(lossGradientMatrix)
	
	local numberOfData = #lossGradientMatrix * #lossGradientMatrix[1]

	self:gradientDescent(lossFunctionGradientMatrixArray, numberOfData)

	if (clearAllMatrices) then 

		self.userItemMatrix = nil 

		self.lossFunctionGradientMatrixArray = nil

	end

end

function SimonFunkMatrixFactorizationModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	local NewSimonFunkMatrixFactorizationModel = MatrixFactorizationBaseModel.new(parameterDictionary)

	setmetatable(NewSimonFunkMatrixFactorizationModel, SimonFunkMatrixFactorizationModel)
	
	NewSimonFunkMatrixFactorizationModel:setName("SimonFunkMatrixFactorization")
	
	local learningRate = parameterDictionary.learningRate or defaultLearningRate

	NewSimonFunkMatrixFactorizationModel.costFunction = parameterDictionary.costFunction or defaultCostFunction
	
	NewSimonFunkMatrixFactorizationModel.userLearningRate = parameterDictionary.userLearningRate or learningRate
	
	NewSimonFunkMatrixFactorizationModel.itemLearningRate = parameterDictionary.itemLearningRate or learningRate

	NewSimonFunkMatrixFactorizationModel.UserOptimizer = parameterDictionary.UserOptimizer
	
	NewSimonFunkMatrixFactorizationModel.ItemOptimizer = parameterDictionary.ItemOptimizer

	NewSimonFunkMatrixFactorizationModel.UserRegularizer = parameterDictionary.UserRegularizer
	
	NewSimonFunkMatrixFactorizationModel.ItemRegularizer = parameterDictionary.ItemRegularizer

	return NewSimonFunkMatrixFactorizationModel

end

function SimonFunkMatrixFactorizationModel:setUserOptimizer(UserOptimizer)

	self.UserOptimizer = UserOptimizer

end

function SimonFunkMatrixFactorizationModel:setItemOptimizer(ItemOptimizer)

	self.ItemOptimizer = ItemOptimizer

end

function SimonFunkMatrixFactorizationModel:setUserRegularizer(UserRegularizer)

	self.UserRegularizer = UserRegularizer

end

function SimonFunkMatrixFactorizationModel:setItemRegularizer(ItemRegularizer)

	self.ItemRegularizer = ItemRegularizer

end

function SimonFunkMatrixFactorizationModel:train(userItemDictionaryDictionary)
	
	local lossFunctionGradientFunctionToApply = lossFunctionGradientList[self.costFunction]

	if (not lossFunctionGradientFunctionToApply) then error("Invalid cost function.") end
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local latentFactorCount = self.latentFactorCount

	local UserOptimizer = self.UserOptimizer
	
	local ItemOptimizer = self.ItemOptimizer
	
	local userItemMatrix, userItemMaskMatrix, numberOfUserIDsAdded, numberOfItemIDsAdded = self:processUserItemDictionaryDictionary(userItemDictionaryDictionary)
	
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

		local hypothesisMatrix = self:calculateHypothesisMatrix(userItemMatrix, true)

		cost = self:calculateCostWhenRequired(numberOfIterations, function()

			return self:calculateCost(hypothesisMatrix, userItemMatrix, userItemMaskMatrix)

		end)

		if (cost) then 

			table.insert(costArray, cost)

			self:printNumberOfIterationsAndCost(numberOfIterations, cost)

		end

		local lossGradientMatrix = AqwamTensorLibrary:applyFunction(lossFunctionGradientFunctionToApply, hypothesisMatrix, userItemMatrix)
		
		lossGradientMatrix = AqwamTensorLibrary:multiply(lossGradientMatrix, userItemMaskMatrix)

		self:update(lossGradientMatrix, true)

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

function SimonFunkMatrixFactorizationModel:predict(userIDVector, returnOriginalOutput)
	
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

return SimonFunkMatrixFactorizationModel
