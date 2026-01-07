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

local MatrixFactorizationBaseModel = {}

MatrixFactorizationBaseModel.__index = MatrixFactorizationBaseModel

setmetatable(MatrixFactorizationBaseModel, GradientMethodBaseModel)

local defaultLatentFactorCount = 1

local function insertIDsToArray(IDArray, IDDictionary)

	local IDArrayLength = #IDArray

	local numberOfNewIDsAdded = 0

	local isIDExist

	for ID in pairs(IDDictionary) do

		isIDExist = false

		for i, storedID in ipairs(IDArray) do

			isIDExist = (storedID == ID)

			if (isIDExist) then break end

		end

		if (not isIDExist) then

			IDArrayLength = IDArrayLength + 1

			IDArray[IDArrayLength] = ID

			numberOfNewIDsAdded = numberOfNewIDsAdded + 1

		end

	end

	return IDArray, numberOfNewIDsAdded
	
end


function MatrixFactorizationBaseModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewMatrixFactorizationBaseModel = GradientMethodBaseModel.new(parameterDictionary)
	
	setmetatable(NewMatrixFactorizationBaseModel, MatrixFactorizationBaseModel)
	
	NewMatrixFactorizationBaseModel:setName("MatrixFactorizationBaseModel")

	NewMatrixFactorizationBaseModel:setClassName("MatrixFactorizationModel")
	
	NewMatrixFactorizationBaseModel.latentFactorCount = parameterDictionary.latentFactorCount or defaultLatentFactorCount
	
	NewMatrixFactorizationBaseModel.userIDArray = parameterDictionary.userIDArray or {}
	
	NewMatrixFactorizationBaseModel.itemIDArray = parameterDictionary.itemIDArray or {}
	
	return NewMatrixFactorizationBaseModel
	
end

function MatrixFactorizationBaseModel:processUserItemDictionaryDictionary(userItemDictionaryDictionary)
	
	local userIDArray, numberOfUserIDsAdded = insertIDsToArray(self.userIDArray, userItemDictionaryDictionary) 
	
	local itemIDArray = self.itemIDArray
	
	local numberOfItemIDsAdded = 0
	
	for userID, userItemDictionary in pairs(userItemDictionaryDictionary) do
		
		local numberOfSubItemIDsAdded = 0
		
		itemIDArray, numberOfSubItemIDsAdded = insertIDsToArray(itemIDArray, userItemDictionaryDictionary) 
		
		numberOfItemIDsAdded = numberOfItemIDsAdded + numberOfSubItemIDsAdded
		
	end
	
	local itemIDArrayLength = #itemIDArray
	
	local userItemMatrix = {}
	
	local userItemDictionary

	local unwrappedUserItemVector
	
	local targetColumnIndex
	
	for i, userID in ipairs(userIDArray) do
		
		userItemDictionary = userItemDictionaryDictionary[userID]
		
		if (userItemDictionary) then
			
			unwrappedUserItemVector = {}
			
			for itemID, value in pairs(userItemDictionary) do
				
				targetColumnIndex = table.find(itemIDArray, itemID)
				
				if (targetColumnIndex) then unwrappedUserItemVector[targetColumnIndex] = value end
				
			end
			
		else
			
			unwrappedUserItemVector = table.create(itemIDArrayLength, 0)
			
		end
		
		userItemMatrix[i] = unwrappedUserItemVector
		
	end
	
	return userItemMatrix, numberOfUserIDsAdded, numberOfItemIDsAdded
	
end

function MatrixFactorizationBaseModel:fetchHighestValueVector(outputMatrix)
	
	local highestValueVector = {}

	local predictedLabelVector = {}

	local highestValue

	local highestIndex

	local value

	for i, unwrappedOutputVector in ipairs(outputMatrix) do

		highestValue = -math.huge

		highestIndex = nil

		for j, outputValue in ipairs(unwrappedOutputVector) do

			if (outputValue > highestValue) then

				highestValue = outputValue

				highestIndex = j

			end

		end

		predictedLabelVector[i] = {highestIndex}

		highestValueVector[i] = {highestValue}

	end

	return predictedLabelVector, highestValueVector

end

return MatrixFactorizationBaseModel
