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

local SimonFunkMatrixFactorizationWithBiasesModel = {}

SimonFunkMatrixFactorizationWithBiasesModel.__index = SimonFunkMatrixFactorizationWithBiasesModel

setmetatable(SimonFunkMatrixFactorizationWithBiasesModel, MatrixFactorizationBaseModel)

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

function SimonFunkMatrixFactorizationWithBiasesModel:calculateCost(hypothesisMatrix, labelMatrix, userItemMaskMatrix)

	if (type(hypothesisMatrix) == "number") then hypothesisMatrix = {{hypothesisMatrix}} end

	local costMatrix = AqwamTensorLibrary:applyFunction(lossFunctionList[self.costFunction], hypothesisMatrix, labelMatrix)
	
	costMatrix = AqwamTensorLibrary:multiply(costMatrix, userItemMaskMatrix)

	local totalCost = AqwamTensorLibrary:sum(costMatrix)
	
	local UserOptimizer = self.UserOptimizer
	
	local ItemOptimizer = self.ItemOptimizer
	
	local UserBiasRegularizer = self.UserBiasRegularizer

	local ItemBiasRegularizer = self.ItemBiasRegularizer

	local AverageRegularizer = self.AverageRegularizer
	
	local ModelParameters = self.ModelParameters or {}

	if (UserOptimizer) then totalCost = totalCost + UserOptimizer:calculateCost(ModelParameters[1]) end
	
	if (ItemOptimizer) then totalCost = totalCost + ItemOptimizer:calculateCost(ModelParameters[2]) end
	
	if (UserBiasRegularizer) then totalCost = totalCost + UserBiasRegularizer:calculateCost(ModelParameters[3]) end
	
	if (ItemBiasRegularizer) then totalCost = totalCost + ItemBiasRegularizer:calculateCost(ModelParameters[4]) end
	
	if (AverageRegularizer) then totalCost = totalCost + AverageRegularizer:calculateCost(ModelParameters[5]) end

	local averageCost = totalCost / (#labelMatrix * #labelMatrix[1])

	return averageCost

end

function SimonFunkMatrixFactorizationWithBiasesModel:calculateHypothesisMatrix(userItemMatrix, saveUserItemMatrix)
	
	local latentFactorCount = self.latentFactorCount
	
	local ModelParameters = self.ModelParameters or {}
	
	local numberOfUsers = #userItemMatrix
	
	local numberOfItems = #userItemMatrix[1]
	
	local userLatentMatrix = ModelParameters[1] or self:initializeMatrixBasedOnMode({numberOfUsers, latentFactorCount})

	local itemLatentMatrix = ModelParameters[2] or self:initializeMatrixBasedOnMode({latentFactorCount, numberOfItems})
	
	local userBiasVector = ModelParameters[3] or self:initializeMatrixBasedOnMode({numberOfUsers, 1})

	local itemBiasMatrix = ModelParameters[4] or self:initializeMatrixBasedOnMode({1, numberOfItems})
	
	local averageMatrix = ModelParameters[5] or self:initializeMatrixBasedOnMode({1, 1})

	local hypothesisMatrix = AqwamTensorLibrary:dotProduct(userLatentMatrix, itemLatentMatrix)
	
	hypothesisMatrix = AqwamTensorLibrary:add(hypothesisMatrix, userBiasVector, itemBiasMatrix, averageMatrix)
	
	self.ModelParameters = {userLatentMatrix, itemLatentMatrix, userBiasVector, itemBiasMatrix, averageMatrix}

	if (saveUserItemMatrix) then self.userItemMatrix = userItemMatrix end

	return hypothesisMatrix

end

function SimonFunkMatrixFactorizationWithBiasesModel:calculateLossFunctionGradientVector(lossFunctionGradientMatrix)

	if (type(lossFunctionGradientMatrix) == "number") then lossFunctionGradientMatrix = {{lossFunctionGradientMatrix}} end
	
	local ModelParameters = self.ModelParameters

	local userLatentMatrix = ModelParameters[1]

	local itemLatentMatrix = ModelParameters[2]
	
	local userBiasVector = ModelParameters[3]

	local itemBiasVector = ModelParameters[4]

	local averageMatrix = ModelParameters[5]

	local userLossFunctionGradientMatrix = AqwamTensorLibrary:dotProduct(lossFunctionGradientMatrix, AqwamTensorLibrary:transpose(itemLatentMatrix))

	local itemLossFunctionGradientMatrix = AqwamTensorLibrary:dotProduct(AqwamTensorLibrary:transpose(userLatentMatrix), lossFunctionGradientMatrix)
	
	local userBiasLossFunctionGradientVector = AqwamTensorLibrary:sum(lossFunctionGradientMatrix, 2)
	
	local itemBiasLossFunctionGradientVector = AqwamTensorLibrary:sum(lossFunctionGradientMatrix, 1)
	
	local averageLossFunctionGradientMatrix = {{AqwamTensorLibrary:sum(lossFunctionGradientMatrix)}}
	
	local lossFunctionGradientMatrixArray = {userLossFunctionGradientMatrix, itemLossFunctionGradientMatrix, userBiasLossFunctionGradientVector, itemBiasLossFunctionGradientVector, averageLossFunctionGradientMatrix}

	if (self.areGradientsSaved) then self.lossFunctionGradientMatrixArray = lossFunctionGradientMatrixArray end

	return lossFunctionGradientMatrixArray

end

function SimonFunkMatrixFactorizationWithBiasesModel:gradientDescent(lossFunctionGradientMatrixArray, numberOfData)
	
	local UserRegularizer = self.UserRegularizer

	local ItemRegularizer = self.ItemRegularizer
	
	local UserBiasRegularizer = self.UserBiasRegularizer

	local ItemBiasRegularizer = self.ItemBiasRegularizer
	
	local AverageRegularizer = self.AverageRegularizer

	local UserOptimizer = self.UserOptimizer

	local ItemOptimizer = self.ItemOptimizer
	
	local UserBiasOptimizer = self.UserBiasOptimizer

	local ItemBiasOptimizer = self.ItemBiasOptimizer
	
	local AverageOptimizer = self.AverageOptimizer

	local userLearningRate = self.userLearningRate

	local itemLearningRate = self.itemLearningRate
	
	local userBiasLearningRate = self.userBiasLearningRate

	local itemBiasLearningRate = self.itemBiasLearningRate
	
	local averageLearningRate = self.averageLearningRate
	
	local ModelParameters = self.ModelParameters
	
	local userLatentMatrix = ModelParameters[1]

	local itemLatentMatrix = ModelParameters[2]
	
	local userBiasVector = ModelParameters[3]

	local itemBiasMatrix = ModelParameters[4]

	local averageMatrix = ModelParameters[5]
	
	local userLatentLossFunctionGradientMatrix = lossFunctionGradientMatrixArray[1]
	
	local itemLatentLossFunctionGradientMatrix = lossFunctionGradientMatrixArray[2]
	
	local userBiasLossFunctionGradientVector = lossFunctionGradientMatrixArray[3]

	local itemBiasLossFunctionGradientVector = lossFunctionGradientMatrixArray[4]
	
	local averageLossFunctionGradientMatrix = lossFunctionGradientMatrixArray[5]
	
	if (UserRegularizer) then

		local userRegularizationGradients = UserRegularizer:calculate(userLatentMatrix)

		userLatentLossFunctionGradientMatrix = AqwamTensorLibrary:add(userLatentLossFunctionGradientMatrix, userRegularizationGradients)

	end
	
	if (ItemRegularizer) then

		local itemRegularizationGradients = ItemRegularizer:calculate(itemLatentMatrix)

		itemLatentLossFunctionGradientMatrix = AqwamTensorLibrary:add(itemLatentLossFunctionGradientMatrix, itemRegularizationGradients)

	end
	
	if (UserBiasRegularizer) then

		local userBiasRegularizationGradients = UserBiasRegularizer:calculate(userBiasVector)

		userBiasLossFunctionGradientVector = AqwamTensorLibrary:add(userBiasLossFunctionGradientVector, userBiasRegularizationGradients)

	end
	
	if (ItemBiasRegularizer) then

		local itemBiasRegularizationGradients = ItemBiasRegularizer:calculate(itemBiasMatrix)

		itemBiasLossFunctionGradientVector = AqwamTensorLibrary:add(itemBiasLossFunctionGradientVector, itemBiasRegularizationGradients)

	end
	
	if (AverageRegularizer) then

		local averageRegularizationGradients = AverageRegularizer:calculate(averageMatrix)

		averageLossFunctionGradientMatrix = AqwamTensorLibrary:add(averageLossFunctionGradientMatrix, averageRegularizationGradients)

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
	
	if (UserBiasOptimizer) then 

		userBiasLossFunctionGradientVector = UserBiasOptimizer:calculate(userBiasLearningRate, userBiasLossFunctionGradientVector, userBiasVector) 

	else

		userBiasLossFunctionGradientVector = AqwamTensorLibrary:multiply(userBiasLearningRate, userBiasLossFunctionGradientVector)

	end
	
	if (ItemBiasOptimizer) then 

		itemBiasLossFunctionGradientVector = ItemBiasOptimizer:calculate(itemBiasLearningRate, itemBiasLossFunctionGradientVector, itemBiasMatrix) 

	else

		itemBiasLossFunctionGradientVector = AqwamTensorLibrary:multiply(itemBiasLearningRate, itemBiasLossFunctionGradientVector)

	end
	
	if (AverageOptimizer) then 

		averageLossFunctionGradientMatrix = AverageOptimizer:calculate(averageLearningRate, averageLossFunctionGradientMatrix, averageMatrix) 

	else

		averageLossFunctionGradientMatrix = AqwamTensorLibrary:multiply(averageLearningRate, averageLossFunctionGradientMatrix)

	end
	
	userLatentLossFunctionGradientMatrix = AqwamTensorLibrary:divide(userLatentLossFunctionGradientMatrix, numberOfData)
	
	itemLatentLossFunctionGradientMatrix = AqwamTensorLibrary:divide(itemLatentLossFunctionGradientMatrix, numberOfData)
	
	userBiasLossFunctionGradientVector = AqwamTensorLibrary:divide(userBiasLossFunctionGradientVector, numberOfData)

	itemBiasLossFunctionGradientVector = AqwamTensorLibrary:divide(itemBiasLossFunctionGradientVector, numberOfData)
	
	averageLossFunctionGradientMatrix = AqwamTensorLibrary:divide(averageLossFunctionGradientMatrix, numberOfData)
	
	userLatentMatrix = AqwamTensorLibrary:subtract(userLatentMatrix, userLatentLossFunctionGradientMatrix)
	
	itemLatentMatrix = AqwamTensorLibrary:subtract(itemLatentMatrix, itemLatentLossFunctionGradientMatrix)
	
	userBiasVector = AqwamTensorLibrary:subtract(userBiasVector, userBiasLossFunctionGradientVector)

	itemBiasMatrix = AqwamTensorLibrary:subtract(itemBiasMatrix, itemBiasLossFunctionGradientVector)
	
	averageMatrix = AqwamTensorLibrary:subtract(averageMatrix, averageLossFunctionGradientMatrix)

	self.ModelParameters = {userLatentMatrix, itemLatentMatrix, userBiasVector, itemBiasMatrix, averageLossFunctionGradientMatrix}

end

function SimonFunkMatrixFactorizationWithBiasesModel:update(lossGradientMatrix, clearAllMatrices)

	if (type(lossGradientMatrix) == "number") then lossGradientMatrix = {{lossGradientMatrix}} end

	local lossFunctionGradientMatrixArray = self:calculateLossFunctionGradientVector(lossGradientMatrix)
	
	local numberOfData = #lossGradientMatrix * #lossGradientMatrix[1]

	self:gradientDescent(lossFunctionGradientMatrixArray, numberOfData)

	if (clearAllMatrices) then 

		self.userItemMatrix = nil 

		self.lossFunctionGradientMatrixArray = nil

	end

end

function SimonFunkMatrixFactorizationWithBiasesModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	local NewSimonFunkMatrixFactorizationWithBiasesModel = MatrixFactorizationBaseModel.new(parameterDictionary)

	setmetatable(NewSimonFunkMatrixFactorizationWithBiasesModel, SimonFunkMatrixFactorizationWithBiasesModel)
	
	NewSimonFunkMatrixFactorizationWithBiasesModel:setName("SimonFunkMatrixFactorizationWithBiases")
	
	local learningRate = parameterDictionary.learningRate or defaultLearningRate

	NewSimonFunkMatrixFactorizationWithBiasesModel.costFunction = parameterDictionary.costFunction or defaultCostFunction
	
	NewSimonFunkMatrixFactorizationWithBiasesModel.userLearningRate = parameterDictionary.userLearningRate or learningRate
	
	NewSimonFunkMatrixFactorizationWithBiasesModel.itemLearningRate = parameterDictionary.itemLearningRate or learningRate
	
	NewSimonFunkMatrixFactorizationWithBiasesModel.userBiasLearningRate = parameterDictionary.userBiasLearningRate or learningRate
	
	NewSimonFunkMatrixFactorizationWithBiasesModel.itemBiasLearningRate = parameterDictionary.itemBiasLearningRate or learningRate

	NewSimonFunkMatrixFactorizationWithBiasesModel.averageLearningRate = parameterDictionary.averageLearningRate or learningRate

	NewSimonFunkMatrixFactorizationWithBiasesModel.UserOptimizer = parameterDictionary.UserOptimizer
	
	NewSimonFunkMatrixFactorizationWithBiasesModel.ItemOptimizer = parameterDictionary.ItemOptimizer
	
	NewSimonFunkMatrixFactorizationWithBiasesModel.UserBiasOptimizer = parameterDictionary.UserBiasOptimizer

	NewSimonFunkMatrixFactorizationWithBiasesModel.ItemBiasOptimizer = parameterDictionary.ItemBiasOptimizer
	
	NewSimonFunkMatrixFactorizationWithBiasesModel.AverageOptimizer = parameterDictionary.AverageOptimizer

	NewSimonFunkMatrixFactorizationWithBiasesModel.UserRegularizer = parameterDictionary.UserRegularizer
	
	NewSimonFunkMatrixFactorizationWithBiasesModel.ItemRegularizer = parameterDictionary.ItemRegularizer
	
	NewSimonFunkMatrixFactorizationWithBiasesModel.UserBiasRegularizer = parameterDictionary.UserBiasRegularizer

	NewSimonFunkMatrixFactorizationWithBiasesModel.ItemBiasRegularizer = parameterDictionary.ItemBiasRegularizer
	
	NewSimonFunkMatrixFactorizationWithBiasesModel.AverageRegularizer = parameterDictionary.AverageRegularizer

	return NewSimonFunkMatrixFactorizationWithBiasesModel

end

function SimonFunkMatrixFactorizationWithBiasesModel:setUserOptimizer(UserOptimizer)

	self.UserOptimizer = UserOptimizer

end

function SimonFunkMatrixFactorizationWithBiasesModel:setItemOptimizer(ItemOptimizer)

	self.ItemOptimizer = ItemOptimizer

end

function SimonFunkMatrixFactorizationWithBiasesModel:setUserRegularizer(UserRegularizer)

	self.UserRegularizer = UserRegularizer

end

function SimonFunkMatrixFactorizationWithBiasesModel:setItemRegularizer(ItemRegularizer)

	self.ItemRegularizer = ItemRegularizer

end

function SimonFunkMatrixFactorizationWithBiasesModel:train(userItemDictionaryDictionary)
	
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
	
	local userBiasMatrix = ModelParameters[3]

	local itemBiasMatrix = ModelParameters[4]
	
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
	
	if (numberOfUserIDsAdded >= 1) and (userBiasMatrix) then

		local userBiasSubMatrix = self:initializeMatrixBasedOnMode({numberOfUserIDsAdded, 1})

		userBiasMatrix = AqwamTensorLibrary:concatenate(userBiasMatrix, userBiasSubMatrix, 1)

		ModelParameters[3] = userBiasMatrix

	end
	
	if (numberOfItemIDsAdded >= 1) and (itemBiasMatrix) then

		local itemBiasSubMatrix = self:initializeMatrixBasedOnMode({1, numberOfItemIDsAdded})

		itemBiasMatrix = AqwamTensorLibrary:concatenate(itemBiasMatrix, itemBiasSubMatrix, 2)

		ModelParameters[4] = itemBiasMatrix

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

function SimonFunkMatrixFactorizationWithBiasesModel:predict(userIDVector, returnOriginalOutput)
	
	local storedUserIDArray = self.userIDArray

	local storedItemIDArray = self.itemIDArray
	
	local ModelParameters = self.ModelParameters or {}

	local userLatentMatrix = ModelParameters[1]

	local itemLatentMatrix = ModelParameters[2]
	
	local userBiasVector = ModelParameters[3]

	local itemBiasVector = ModelParameters[4]

	local averageMatrix = ModelParameters[5]
	
	if (userIDVector) then
		
		local userLatentSubMatrix = {}
		
		local userBiasSubVector = {}
		
		for i, unwrappedUserIDVector in ipairs(userIDVector) do
			
			local targetIndex = table.find(storedUserIDArray, unwrappedUserIDVector[1])
			
			if (targetIndex) then 
				
				userLatentSubMatrix[i] = userLatentMatrix[targetIndex] 
				
				userBiasSubVector[i] = userBiasVector[targetIndex] 
				
			end
			
		end
		
		userLatentMatrix = userLatentSubMatrix
		
		userBiasVector = userBiasSubVector
		
	end
	
	local predictedMatrix = AqwamTensorLibrary:dotProduct(userLatentMatrix, itemLatentMatrix)
	
	predictedMatrix = AqwamTensorLibrary:add(predictedMatrix, userBiasVector, itemBiasVector, averageMatrix)

	if (returnOriginalOutput) then return predictedMatrix end
	
	return self:fetchHighestValueVector(predictedMatrix)

end

return SimonFunkMatrixFactorizationWithBiasesModel
