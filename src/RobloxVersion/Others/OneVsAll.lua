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

local DataPredictLibrary = script.Parent.Parent

local IterativeMethodBaseModel = require(DataPredictLibrary.Models.IterativeMethodBaseModel)

local Models = DataPredictLibrary.Models

local Optimizers = DataPredictLibrary.Optimizers

local Regularizers = require(DataPredictLibrary.Regularizers)

local AqwamTensorLibrary = require(DataPredictLibrary.AqwamTensorLibraryLinker.Value)

OneVsAll = {}

OneVsAll.__index = OneVsAll

setmetatable(OneVsAll, IterativeMethodBaseModel)

local defaultModelName = "LogisticRegression"

local defaultNumberOfClasses = 2

local defaultMaximumNumberOfIterations = 500

function OneVsAll.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations
	
	local NewOneVsAll = IterativeMethodBaseModel.new(parameterDictionary)
	
	setmetatable(NewOneVsAll, OneVsAll)
	
	NewOneVsAll:setName("OneVsAll")
	
	NewOneVsAll:setClassName("OneVsAll")
	
	NewOneVsAll.numberOfClasses = parameterDictionary.numberOfClasses or defaultNumberOfClasses
	
	NewOneVsAll.useNegativeOneBinaryLabel = NewOneVsAll:getValueOrDefaultValue(parameterDictionary.useNegativeOneBinaryLabel, false)
	
	NewOneVsAll.ClassesList = parameterDictionary.ClassesList or {}
	
	NewOneVsAll.ModelArray = parameterDictionary.ModelArray or {}
	
	return NewOneVsAll
	
end

function OneVsAll:generateModel(parameterDictionary)
	
	local modelName = parameterDictionary.modelName
	
	if (not modelName) then error("No model name.") end
	
	parameterDictionary = parameterDictionary or {}

	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or 1

	parameterDictionary.isOutputPrinted = self:getValueOrDefaultValue(parameterDictionary.isOutputPrinted, false)
	
	local ModelArray = self.ModelArray
	
	local SelectedModel = require(Models[modelName])

	for i = 1, self.numberOfClasses, 1 do

		local ModelObject = SelectedModel.new(parameterDictionary)

		table.insert(ModelArray, ModelObject)

	end

	self.ModelArray = ModelArray
	
end

function OneVsAll:setModel(parameterDictionary)
	
	local ModelArray = self.ModelArray
	
	if (#ModelArray == 0) then 
		
		self:generateModel(parameterDictionary) 
		
	else
		
		for parameterKey, parameterValue in parameterDictionary do

			for _, Model in ipairs(ModelArray) do Model[parameterKey] = parameterValue end

		end
		
	end
	
end

function OneVsAll:setOptimizer(parameterDictionary)
	
	if (not parameterDictionary) then return end
	
	local ModelArray = self.ModelArray
	
	if (#ModelArray == 0) then error("No model.") end
	
	local optimizerName = parameterDictionary.optimizerName
	
	if (not optimizerName) then error("No optimizer name.") end
	
	local SelectedOptimizer = require(Optimizers[optimizerName])
		
	for m, Model in ipairs(ModelArray) do 

		local success = pcall(function() 
				
			local OptimizerObject = SelectedOptimizer.new(parameterDictionary)

			Model:setOptimizer(OptimizerObject)
				
		end)

		if (not success) then warn("Model " .. m .. " does not have setOptimizer() function. No optimizer have been set to the model.") end

	end
		
end

function OneVsAll:setRegularizer(parameterDictionary)
	
	if (not parameterDictionary) then return end
	
	local ModelArray = self.ModelArray
	
	if (#ModelArray == 0) then error("No model.") end
	
	local regularizerName = parameterDictionary.regularizerName
	
	if (not regularizerName) then error("No regularizer name.") end
	
	local SelectedRegularizer = require(Regularizers[regularizerName])
	
	local RegularizerObject = SelectedRegularizer.new(parameterDictionary)
	
	for m, Model in ipairs(ModelArray) do 
		
		local success = pcall(function() Model:setRegularizer(RegularizerObject) end)
		
		if (not success) then warn("Model " .. m .. " does not have setRegularizer() function. No regularizer have been set to the model.") end
	
	end
	
end

function OneVsAll:setClassesList(classesList)

	self.ClassesList = classesList

end

function OneVsAll:getClassesList()

	return self.ClassesList

end

local function checkIfAnyLabelVectorIsNotRecognized(labelVector, ClassesList)

	for i = 1, #labelVector, 1 do

		if table.find(ClassesList, labelVector[i][1]) then continue end

		return true

	end

	return false

end

local function createClassesList(labelVector)

	local ClassesList = {}

	local value

	for i = 1, #labelVector, 1 do

		value = labelVector[i][1]

		if not table.find(ClassesList, value) then

			table.insert(ClassesList, value)

		end

	end

	return ClassesList

end

function OneVsAll:processLabelVector(labelVector)
	
	local ClassesList = self.ClassesList

	if (#ClassesList == 0) then

		ClassesList = createClassesList(labelVector)

		table.sort(ClassesList, function(a,b) return a < b end)

	else

		if checkIfAnyLabelVectorIsNotRecognized(labelVector, ClassesList) then error("A value does not exist in the classes list is present in the label vector") end

	end
	
	self.ClassesList = ClassesList

end

local function convertToBinaryLabelVector(labelVector, selectedClass, useNegativeOneBinaryLabel)

	local numberOfRows = #labelVector

	local newLabelVector = AqwamTensorLibrary:createTensor({numberOfRows, 1}, true)

	for row = 1, numberOfRows, 1 do

		if (labelVector[row][1] == selectedClass) then

			newLabelVector[row][1] = 1

		else

			newLabelVector[row][1] = (useNegativeOneBinaryLabel and -1) or 0

		end

	end

	return newLabelVector

end

function OneVsAll:train(featureMatrix, labelVector)
	
	local ModelArray = self.ModelArray
	
	if (#ModelArray == 0) then self:generateModel() end
	
	self:processLabelVector(labelVector)
	
	local ClassesList = self.ClassesList
	
	if (#ModelArray ~= #ClassesList) then error("The number of models does not match with number of classes.") end
	
	local useNegativeOneBinaryLabel = self.useNegativeOneBinaryLabel
	
	local binaryLabelVectorTable = {}
	
	for i, class in ipairs(ClassesList) do

		local binaryLabelVector = convertToBinaryLabelVector(labelVector, class, useNegativeOneBinaryLabel)

		table.insert(binaryLabelVectorTable, binaryLabelVector)

	end
	
	local ModelArray = self.ModelArray
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local isOutputPrinted = self.isOutputPrinted
	
	local numberOfIterations = 0
	
	local costArray = {}
	
	local modelCostArray
	
	repeat
		
		local totalCost = 0
		
		for m, Model in ipairs(ModelArray) do
			
			local binaryLabelVector = binaryLabelVectorTable[m]

			modelCostArray = Model:train(featureMatrix, binaryLabelVector)

			totalCost = totalCost + modelCostArray[#modelCostArray]

		end
		
		numberOfIterations = numberOfIterations + 1
		
		table.insert(costArray, totalCost)
		
		self:printCostAndNumberOfIterations(totalCost, numberOfIterations)
				
	until (numberOfIterations >= maximumNumberOfIterations) or self:checkIfTargetCostReached(totalCost) or self:checkIfConverged(totalCost)
	
	return costArray
	
end

function OneVsAll:getBestPrediction(featureVector)
	
	local selectedModelNumber = 0
	
	local highestValue = -math.huge
	
	for m, Model in ipairs(self.ModelArray) do 

		local allOutputVector = Model:predict(featureVector, true)
		
		if (typeof(allOutputVector) == "number") then allOutputVector = {{allOutputVector}} end

		local dimensionIndexArray, value = AqwamTensorLibrary:findMaximumValueDimensionIndexArray(allOutputVector)

		if (dimensionIndexArray) then
			
			if (value > highestValue) then
				
				selectedModelNumber = m

				highestValue = value
				
			end
			
		end

	end
	
	return selectedModelNumber, highestValue
	
end

function OneVsAll:predict(featureMatrix)
	
	if (#self.ModelArray == 0) then error("No model set.") end
	
	local numberOfData = #featureMatrix
	
	local selectedModelNumberVector = AqwamTensorLibrary:createTensor({numberOfData, 1})
	
	local highestValueVector = AqwamTensorLibrary:createTensor({numberOfData, 1})
	
	for i = 1, #featureMatrix, 1 do
		
		local featureVector = {featureMatrix[i]}
		
		local selectedModelNumber, highestValue = self:getBestPrediction(featureVector)
		
		selectedModelNumberVector[i][1] = self.ClassesList[selectedModelNumber]
		
		highestValueVector[i][1] = highestValue
		
	end
	
	return selectedModelNumberVector, highestValueVector
	
end

function OneVsAll:getModelParametersArray(doNotDeepCopy)
	
	if (#self.ModelArray == 0) then error("No model set.") end
	
	local ModelParametersArray = {}
	
	for _, Model in ipairs(self.ModelArray) do 
		
		local ModelParameters = Model:getModelParameters(doNotDeepCopy)
		
		table.insert(ModelParametersArray, ModelParameters) 
		
	end
	
	return ModelParametersArray
	
end

function OneVsAll:setModelParametersArray(ModelParametersArray, doNotDeepCopy)
	
	if (#self.ModelArray == 0) then error("No model set.") end
	
	if (ModelParametersArray == nil) then return nil end
	
	if (#ModelParametersArray ~= #self.ModelArray) then error("The number of model parameters does not match with the number of models!") end
	
	for m, Model in ipairs(self.ModelArray) do 
		
		local ModelParameters = ModelParametersArray[m]

		Model:setModelParameters(ModelParameters, doNotDeepCopy)

	end
	
end

function OneVsAll:clearModelParameters()
	
	if (#self.ModelArray == 0) then error("No model set.") end
	
	for _, Model in ipairs(self.ModelArray) do Model:clearModelParameters() end

end

return OneVsAll
