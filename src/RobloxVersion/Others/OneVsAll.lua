local DataPredictLibrary = script.Parent.Parent

local Models = DataPredictLibrary.Models

local Optimizers = DataPredictLibrary.Optimizers

local Regularization = require(DataPredictLibrary.Others.Regularization)

local AqwamMatrixLibrary = require(DataPredictLibrary.AqwamMatrixLibraryLinker.Value)

OneVsAll = {}

OneVsAll.__index = OneVsAll

local defaultMaxNumberOfIterations = 500

local defaultTotalTargetCost = 0

function OneVsAll.new(maxNumberOfIterations, useNegativeOneBinaryLabel, totalTargetCost)
	
	local NewOneVsAll = {}
	
	setmetatable(NewOneVsAll, OneVsAll)
	
	NewOneVsAll.maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations
	
	NewOneVsAll.useNegativeOneBinaryLabel = useNegativeOneBinaryLabel or false
	
	NewOneVsAll.totalTargetCost = totalTargetCost or defaultTotalTargetCost
	
	NewOneVsAll.IsOutputPrinted = true
	
	NewOneVsAll.ModelsArray = {}
	
	NewOneVsAll.OptimizersArray = {}
	
	NewOneVsAll.ClassesList = {}
	
	return NewOneVsAll
	
end

function OneVsAll:getBooleanOrDefaultOption(boolean, defaultBoolean)

	if (type(boolean) == "nil") then return defaultBoolean end

	return boolean

end

function OneVsAll:checkIfModelsSet()
	
	local numberOfModels = #self.ModelsArray

	if (numberOfModels == 0) then error("No models set!") end
	
end

function OneVsAll:setParameters(maxNumberOfIterations, useNegativeOneBinaryLabel, totalTargetCost)
	
	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations
	
	self.useNegativeOneBinaryLabel = self:getBooleanOrDefaultOption(useNegativeOneBinaryLabel, self.useNegativeOneBinaryLabel)
	
	self.totalTargetCost = totalTargetCost or self.totalTargetCost 
	
end

function OneVsAll:setModels(modelName, numberOfClasses)
	
	local ModelObject
	
	local SelectedModel
	
	local ModelsArray = {}
	
	local isNameAdded = (typeof(modelName) == "string")
	
	if isNameAdded then  SelectedModel = require(Models[modelName]) end
	
	for i = 1, numberOfClasses, 1 do

		if (isNameAdded == nil) then continue end

		ModelObject = SelectedModel.new(1)
			
		ModelObject:setPrintOutput(false)
		
		table.insert(ModelsArray, ModelObject)

	end
	
	self.ModelsArray = ModelsArray
	
end

function OneVsAll:setOptimizer(optimizerName, ...)
	
	self:checkIfModelsSet()

	local OptimizerObject
	
	local isNameAdded = (typeof(optimizerName) == "string")
	
	local SelectedOptimizer
	
	if isNameAdded then SelectedOptimizer = require(Optimizers[optimizerName]) end
	
	local success = pcall(function()
		
		self.ModelsArray[1]:setOptimizer() 
		
	end)
	
	if (success == false) then 
		
		warn("The model do not have setOptimizer() function. No optimizer objects have been added.") 
		
		return nil
		
	end
	
	for _, Model in ipairs(self.ModelsArray) do 

		if SelectedOptimizer then

			OptimizerObject = SelectedOptimizer.new(...)

		end

		Model:setOptimizer(OptimizerObject) 

	end
	
end

function OneVsAll:setRegularization(lambda, regularizationMode, hasBias)
	
	self:checkIfModelsSet()
	
	local RegularizationObject

	if lambda or regularizationMode or hasBias then
		
		RegularizationObject = Regularization.new(lambda, regularizationMode, hasBias)
	
	else
		
		RegularizationObject = nil
		
	end
	
	local success = pcall(function()
		
		for _, Model in ipairs(self.ModelsArray) do Model:setRegularization(RegularizationObject) end
		
	end)
	
	if (success == false) then warn("The model do not have setRegularization() function. No regularization objects have been added.") end
	
end

function OneVsAll:setModelsSettings(...)
	
	self:checkIfModelsSet()
	
	for _, Model in ipairs(self.ModelsArray) do Model:setParameters(...) end
	
end

function OneVsAll:setClassesList(classesList)

	self.ClassesList = classesList

end

function OneVsAll:getClassesList()

	return self.ClassesList

end

local function checkIfAnyLabelVectorIsNotRecognized(labelVector, classesList)

	for i = 1, #labelVector, 1 do

		if table.find(classesList, labelVector[i][1]) then continue end

		return true

	end

	return false

end

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

function OneVsAll:processLabelVector(labelVector)

	if (#self.ClassesList == 0) then

		self.ClassesList = createClassesList(labelVector)

		table.sort(self.ClassesList, function(a,b) return a < b end)

	else

		if checkIfAnyLabelVectorIsNotRecognized(labelVector, self.ClassesList) then error("A value does not exist in the classes list is present in the label vector") end

	end

end

local function convertToBinaryLabelVector(labelVector, selectedClass, useNegativeOneBinaryLabel)

	local numberOfRows = #labelVector

	local newLabelVector = AqwamMatrixLibrary:createMatrix(numberOfRows, 1)

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
	
	self:checkIfModelsSet()
	
	self:processLabelVector(labelVector)
	
	if (#self.ModelsArray ~= #self.ClassesList) then error("The number of models does not match with number of classes.") end
	
	local binaryLabelVectorTable = {}
	
	for i, class in ipairs(self.ClassesList) do

		local binaryLabelVector = convertToBinaryLabelVector(labelVector, class, self.useNegativeOneBinaryLabel)

		table.insert(binaryLabelVectorTable, binaryLabelVector)

	end
	
	local costArray = {}
	
	local numberOfIterations = 0
	
	local modelCostArray
	
	repeat
		
		local totalCost = 0
		
		for m, Model in ipairs(self.ModelsArray) do
			
			local binaryLabelVector = binaryLabelVectorTable[m]

			modelCostArray = Model:train(featureMatrix, binaryLabelVector)

			totalCost += modelCostArray[#modelCostArray]

		end
		
		numberOfIterations += 1
		
		table.insert(costArray, totalCost)
		
		if (self.IsOutputPrinted) then print("Iteration: " .. numberOfIterations .. "\t\tCost: " .. totalCost) end
		
	until (numberOfIterations >= self.maxNumberOfIterations) or (totalCost <= self.totalTargetCost)
	
	return costArray
	
end

function OneVsAll:getBestPrediction(featureVector)
	
	local selectedModelNumber = 0
	
	local highestValue = -math.huge
	
	for m, Model in ipairs(self.ModelsArray) do 

		local allOutputVector = Model:predict(featureVector, true)
		
		if (typeof(allOutputVector) == "number") then allOutputVector = {{allOutputVector}} end

		local value, maximumValueIndex = AqwamMatrixLibrary:findMaximumValueInMatrix(allOutputVector)

		if (maximumValueIndex == nil) then continue end

		if (value <= highestValue) then continue end
		
		selectedModelNumber = m

		highestValue = value

	end
	
	return selectedModelNumber, highestValue
	
end

function OneVsAll:predict(featureMatrix)
	
	self:checkIfModelsSet()
	
	local selectedModelNumberVector = AqwamMatrixLibrary:createMatrix(#featureMatrix, 1)
	
	local highestValueVector = AqwamMatrixLibrary:createMatrix(#featureMatrix, 1)
	
	for i = 1, #featureMatrix, 1 do
		
		local featureVector = {featureMatrix[i]}
		
		local selectedModelNumber, highestValue = self:getBestPrediction(featureVector)
		
		selectedModelNumberVector[i][1] = self.ClassesList[selectedModelNumber]
		
		highestValueVector[i][1] = highestValue
		
	end
	
	return selectedModelNumberVector, highestValueVector
	
end

function OneVsAll:getModelParametersArray()
	
	self:checkIfModelsSet()
	
	local ModelParametersArray = {}
	
	for _, Model in ipairs(self.ModelsArray) do 
		
		local ModelParameters = Model:getModelParameters()
		
		table.insert(ModelParametersArray, ModelParameters) 
		
	end
	
	return ModelParametersArray
	
end

function OneVsAll:setModelParametersArray(ModelParametersArray)
	
	self:checkIfModelsSet()
	
	if (ModelParametersArray == nil) then return nil end
	
	if (#ModelParametersArray ~= #self.ModelsArray) then error("The number of model parameters does not match with the number of models!") end
	
	for m, Model in ipairs(self.ModelsArray) do 
		
		local ModelParameters = ModelParametersArray[m]

		Model:setModelParameters(ModelParameters)

	end
	
end

function OneVsAll:clearModelParameters()
	
	self:checkIfModelsSet()
	
	for _, Model in ipairs(self.ModelsArray) do Model:clearModelParameters() end

end

function OneVsAll:setPrintOutput(option) 

	self.IsOutputPrinted = self:getBooleanOrDefaultOption(option, self.IsOutputPrinted)

end

function OneVsAll:setAutoResetOptimizers(option)

	self:checkIfModelsSet()

	for _, Model in ipairs(self.ModelsArray) do Model:setAutoResetOptimizers(option) end

end

function OneVsAll:setNumberOfIterationsToCheckIfConverged(numberOfIterations)
	
	for _, Model in ipairs(self.ModelsArray) do Model:setNumberOfIterationsToCheckIfConverged(numberOfIterations) end
	
end

function OneVsAll:setTargetCost(upperBound, lowerBound)
	

	for _, Model in ipairs(self.ModelsArray) do Model:setTargetCost(upperBound, lowerBound) end
	
end

return OneVsAll
