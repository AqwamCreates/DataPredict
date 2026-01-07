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

local TwoTowerModel = {}

TwoTowerModel.__index = TwoTowerModel

setmetatable(TwoTowerModel, IterativeMethodBaseModel)

local defaultCostFunction = "MeanSquaredError"

local lossFunctionList = {

	["MeanSquaredError"] = function (h, y) return ((h - y)^2) end,

	["MeanAbsoluteError"] = function (h, y) return math.abs(h - y) end,

}

local lossFunctionGradientList = {

	["MeanSquaredError"] = function (h, y) return (2 * (h - y)) end,

	["MeanAbsoluteError"] = function (h, y) return math.sign(h - y) end,

}

function TwoTowerModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewTwoTowerModel = IterativeMethodBaseModel.new(parameterDictionary)
	
	setmetatable(NewTwoTowerModel, TwoTowerModel)
	
	NewTwoTowerModel:setName("TwoTower")
	
	NewTwoTowerModel.costFunction = parameterDictionary.costFunction or defaultCostFunction
	
	NewTwoTowerModel.UserTowerModel = parameterDictionary.UserTowerModel
	
	NewTwoTowerModel.ItemTowerModel = parameterDictionary.ItemTowerModel
	
	return NewTwoTowerModel
	
end

function TwoTowerModel:calculateCost(hypothesisVector, labelVector)

	if (type(hypothesisVector) == "number") then hypothesisVector = {{hypothesisVector}} end

	local costVector = AqwamTensorLibrary:applyFunction(lossFunctionList[self.costFunction], hypothesisVector, labelVector)

	local totalCost = AqwamTensorLibrary:sum(costVector)

	local averageCost = totalCost / (#labelVector * #labelVector[1])

	return averageCost

end

function TwoTowerModel:train(userFeatureMatrix, itemFeatureMatrix, userItemMatrix)

	-- Tower input dimension sizes: (number of users x number of user features), (number of items x number of item features).

	-- Tower output dimension sizes: (number of users x number of embedded user features), (number of items x number of embedded item features).
	
	-- Dot product output dimension size: (number of users x number of items).
	
	local UserTowerModel = self.UserTowerModel
	
	local ItemTowerModel = self.ItemTowerModel
	
	if (not UserTowerModel) then error("No user tower model.") end
	
	if (not ItemTowerModel) then error("No item tower model.") end
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local isOutputPrinted = self.isOutputPrinted
	
	local lossFunctionGradientFunctionToApply = lossFunctionGradientList[self.costFunction]

	if (not lossFunctionGradientFunctionToApply) then error("Invalid cost function.") end
	
	local costArray = {}
	
	local numberOfIterations = 0
	
	local cost

	repeat
		
		self:iterationWait()
		
		numberOfIterations = numberOfIterations + 1
		
		local userEmbeddingMatrix = UserTowerModel:forwardPropagate(userFeatureMatrix, true)

		local itemEmbeddingMatrix = ItemTowerModel:forwardPropagate(itemFeatureMatrix, true)
		
		local transposedItemEmbeddingMatrix = AqwamTensorLibrary:transpose(itemEmbeddingMatrix)

		local similarityMatrix = AqwamTensorLibrary:dotProduct(userEmbeddingMatrix, transposedItemEmbeddingMatrix)
		
		local lossFunctionGradientMatrix = AqwamTensorLibrary:applyFunction(lossFunctionGradientFunctionToApply, similarityMatrix, userItemMatrix)
		
		local userLossGradientMatrix = AqwamTensorLibrary:dotProduct(lossFunctionGradientMatrix, itemEmbeddingMatrix)

		local itemLossGradientMatrix = AqwamTensorLibrary:dotProduct(AqwamTensorLibrary:transpose(userEmbeddingMatrix), lossFunctionGradientMatrix)
		
		local transposedItemLossGradientMatrix = AqwamTensorLibrary:transpose(itemLossGradientMatrix)
		
		UserTowerModel:update(userLossGradientMatrix, true)
		
		ItemTowerModel:update(transposedItemLossGradientMatrix, true)
		
		cost = self:calculateCostWhenRequired(numberOfIterations, function()

			return self:calculateCost(similarityMatrix, userItemMatrix)

		end)

		if (cost) then 

			table.insert(costArray, cost)

			self:printNumberOfIterationsAndCost(numberOfIterations, cost)

		end
		
		if (isOutputPrinted) then print("Iteration: " .. numberOfIterations .. "\t\tCost: " .. cost) end
		
	until (numberOfIterations >= maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)
	
	if (isOutputPrinted) then

		if (cost == math.huge) then warn("The model diverged.") end

		if (cost ~= cost) then warn("The model produced nan (not a number) values.") end

	end
	
	return costArray
	
end

function TwoTowerModel:predict(userFeatureMatrix, itemFeatureMatrix, returnOriginalOutput)
	
	local UserTowerModel = self.UserTowerModel

	local ItemTowerModel = self.ItemTowerModel

	if (not UserTowerModel) then error("No user tower model.") end

	if (not ItemTowerModel) then error("No item tower model.") end
	
	local userEmbeddingMatrix = UserTowerModel:predict(userFeatureMatrix, true)
	
	local itemEmbeddingMatrix = ItemTowerModel:predict(itemFeatureMatrix, true)
	
	local transposedItemEmbeddingMatrix = AqwamTensorLibrary:transpose(itemEmbeddingMatrix)
	
	local similarityMatrix = AqwamTensorLibrary:dotProduct(userEmbeddingMatrix, transposedItemEmbeddingMatrix)
	
	if (returnOriginalOutput) then return similarityMatrix end
	
	local highestValueVector = {}
	
	local predictedLabelVector = {}
	
	local highestValue
	
	local highestIndex
	
	local value
	
	for i, unwrappedSimilarityMatrix in ipairs(similarityMatrix) do
		
		highestValue = -math.huge
		
		highestIndex = nil
		
		for j, similarityValue in ipairs(unwrappedSimilarityMatrix) do

			if (similarityValue > highestValue) then
				
				highestValue = similarityValue
				
				highestIndex = j
				
			end
			
		end
		
		predictedLabelVector[i] = {highestIndex}
		
		highestValueVector[i] = {highestValue}
		
	end
	
	return predictedLabelVector, highestValueVector
	
end

function TwoTowerModel:getUserTowerModel()
	
	return self.UserTowerModel
	
end

function TwoTowerModel:getItemTowerModel()

	return self.ItemTowerModel

end

function TwoTowerModel:setUserTowerModel(UserTowerModel)

	self.UserTowerModel = UserTowerModel

end

function TwoTowerModel:setItemTowerModel(ItemTowerModel)

	self.ItemTowerModel = ItemTowerModel

end

return TwoTowerModel
