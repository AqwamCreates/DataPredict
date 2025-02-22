--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

local Regularization = require("Other_Regularization")

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

OneVsAll = {}

OneVsAll.__index = OneVsAll

local defaultMaximumNumberOfIterations = 500

local defaultTotalTargetCostUpperBound = 0

local defaultTotalTargetCostLowerBound = 0

function OneVsAll.new(maximumNumberOfIterations, useNegativeOneBinaryLabel)
	
	local NewOneVsAll = {}
	
	setmetatable(NewOneVsAll, OneVsAll)
	
	NewOneVsAll.maximumNumberOfIterations = maximumNumberOfIterations or defaultMaximumNumberOfIterations
	
	NewOneVsAll.useNegativeOneBinaryLabel = useNegativeOneBinaryLabel or false
	
	NewOneVsAll.IsOutputPrinted = true
	
	NewOneVsAll.targetTotalCostUpperBound = defaultTotalTargetCostUpperBound

	NewOneVsAll.targetTotalCostLowerBound = defaultTotalTargetCostLowerBound
	
	NewOneVsAll.numberOfIterationsToCheckIfConverged = math.huge
	
	NewOneVsAll.currentNumberOfIterationsToCheckIfConverged = 0
	
	NewOneVsAll.currentCostToCheckForConvergence = nil
	
	NewOneVsAll.ModelArray = {}
	
	NewOneVsAll.OptimizersArray = {}
	
	NewOneVsAll.ClassesList = {}
	
	return NewOneVsAll
	
end

function OneVsAll:getBooleanOrDefaultOption(boolean, defaultBoolean)

	if (type(boolean) == "nil") then return defaultBoolean end

	return boolean

end

function OneVsAll:checkIfModelsSet()
	
	local numberOfModels = #self.ModelArray

	if (numberOfModels == 0) then error("No models set!") end
	
end

function OneVsAll:setParameters(maximumNumberOfIterations, useNegativeOneBinaryLabel)
	
	self.maximumNumberOfIterations = maximumNumberOfIterations or self.maximumNumberOfIterations
	
	self.useNegativeOneBinaryLabel = self:getBooleanOrDefaultOption(useNegativeOneBinaryLabel, self.useNegativeOneBinaryLabel)
	
end

function OneVsAll:setModels(modelName, numberOfClasses)
	
	local ModelObject
	
	local SelectedModel
	
	local ModelArray = {}
	
	local isNameAdded = (typeof(modelName) == "string")
	
	if isNameAdded then SelectedModel = require(Models[modelName]) end
	
	for i = 1, numberOfClasses, 1 do

		if (isNameAdded == nil) then continue end

		ModelObject = SelectedModel.new(1)
			
		ModelObject:setPrintOutput(false)
		
		table.insert(ModelArray, ModelObject)

	end
	
	self.ModelArray = ModelArray
	
end

function OneVsAll:setOptimizer(optimizerName, ...)
	
	self:checkIfModelsSet()

	local OptimizerObject
	
	local isNameAdded = (typeof(optimizerName) == "string")
	
	local SelectedOptimizer
	
	if isNameAdded then SelectedOptimizer = require(Optimizers[optimizerName]) end
	
	local success = pcall(function()
		
		self.ModelArray[1]:setOptimizer() 
		
	end)
	
	if (success == false) then 
		
		warn("The model does not have setOptimizer() function. No optimizer have been added.") 
		
		return nil
		
	end
	
	for _, Model in ipairs(self.ModelArray) do 

		if SelectedOptimizer then

			OptimizerObject = SelectedOptimizer.new(...)

		end

		Model:setOptimizer(OptimizerObject) 

	end
	
end

function OneVsAll:setRegularizer(lambda, regularizationMode, hasBias)
	
	self:checkIfModelsSet()
	
	local RegularizerObject

	if (lambda) or (regularizationMode) or (hasBias) then RegularizerObject = Regularizer.new(lambda, regularizationMode, hasBias) end
	
	local success = pcall(function()
		
		for _, Model in ipairs(self.ModelArray) do Model:setRegularizer(RegularizerObject) end
		
	end)
	
	if (success == false) then warn("The model does not have setRegularizer() function. No regularizer have been added.") end
	
end

function OneVsAll:setModelsSettings(...)
	
	self:checkIfModelsSet()
	
	for _, Model in ipairs(self.ModelArray) do Model:setParameters(...) end
	
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
	
	if (#self.ModelArray ~= #self.ClassesList) then error("The number of models does not match with number of classes.") end
	
	local binaryLabelVectorTable = {}
	
	for i, class in ipairs(self.ClassesList) do

		local binaryLabelVector = convertToBinaryLabelVector(labelVector, class, self.useNegativeOneBinaryLabel)

		table.insert(binaryLabelVectorTable, binaryLabelVector)

	end
	
	local ModelArray = self.ModelArray
	
	local targetTotalCostLowerBound = self.targetTotalCostLowerBound
	
	local targetTotalCostUpperBound = self.targetTotalCostUpperBound
	
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
		
		if (isOutputPrinted) then print("Iteration: " .. numberOfIterations .. "\t\tCost: " .. totalCost) end
		
	until (numberOfIterations >= maximumNumberOfIterations) or ((totalCost >= targetTotalCostLowerBound) and (totalCost <= targetTotalCostUpperBound)) or self:checkIfConverged(totalCost)
	
	return costArray
	
end

function OneVsAll:checkIfConverged(cost)

	if (not cost) then return false end

	if (not self.currentCostToCheckForConvergence) then

		self.currentCostToCheckForConvergence = cost

		return false

	end

	if (self.currentCostToCheckForConvergence ~= cost) then

		self.currentNumberOfIterationsToCheckIfConverged = 1

		self.currentCostToCheckForConvergence = cost

		return false

	end

	if (self.currentNumberOfIterationsToCheckIfConverged < self.numberOfIterationsToCheckIfConverged) then

		self.currentNumberOfIterationsToCheckIfConverged += 1

		return false

	end

	self.currentNumberOfIterationsToCheckIfConverged = 1

	self.currentCostToCheckForConvergence = nil

	return true

end

function OneVsAll:getBestPrediction(featureVector)
	
	local selectedModelNumber = 0
	
	local highestValue = -math.huge
	
	for m, Model in ipairs(self.ModelArray) do 

		local allOutputVector = Model:predict(featureVector, true)
		
		if (typeof(allOutputVector) == "number") then allOutputVector = {{allOutputVector}} end

		local value, maximumValueIndex = AqwamMatrixLibrary:findMaximumValue(allOutputVector)

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

function OneVsAll:getModelParametersArray(doNotDeepCopy)
	
	self:checkIfModelsSet()
	
	local ModelParametersArray = {}
	
	for _, Model in ipairs(self.ModelArray) do 
		
		local ModelParameters = Model:getModelParameters(doNotDeepCopy)
		
		table.insert(ModelParametersArray, ModelParameters) 
		
	end
	
	return ModelParametersArray
	
end

function OneVsAll:setModelParametersArray(ModelParametersArray, doNotDeepCopy)
	
	self:checkIfModelsSet()
	
	if (ModelParametersArray == nil) then return nil end
	
	if (#ModelParametersArray ~= #self.ModelArray) then error("The number of model parameters does not match with the number of models!") end
	
	for m, Model in ipairs(self.ModelArray) do 
		
		local ModelParameters = ModelParametersArray[m]

		Model:setModelParameters(ModelParameters, doNotDeepCopy)

	end
	
end

function OneVsAll:clearModelParameters()
	
	self:checkIfModelsSet()
	
	for _, Model in ipairs(self.ModelArray) do Model:clearModelParameters() end

end

function OneVsAll:setPrintOutput(option) 

	self.isOutputPrinted = self:getBooleanOrDefaultOption(option, self.isOutputPrinted)

end

function OneVsAll:setAutoResetOptimizers(option)

	self:checkIfModelsSet()

	for _, Model in ipairs(self.ModelArray) do Model:setAutoResetOptimizers(option) end

end

function OneVsAll:setNumberOfIterationsToCheckIfConverged(numberOfIterations)
	
	for _, Model in ipairs(self.ModelArray) do Model:setNumberOfIterationsToCheckIfConverged(numberOfIterations) end
	
end

function OneVsAll:setNumberOfIterationsToCheckIfConvergedForOneVsAll(numberOfIterations)

	self.numberOfIterationsToCheckIfConverged = numberOfIterations or self.numberOfIterationsToCheckIfConverged

end

function OneVsAll:setTargetCost(upperBound, lowerBound)
	
	for _, Model in ipairs(self.ModelArray) do Model:setTargetCost(upperBound, lowerBound) end
	
end

function OneVsAll:setTargetTotalCost(upperBound, lowerBound)
	
	self.targetTotalCostUpperBound = upperBound or self.targetTotalCostUpperBound
	
	self.targetTotalCostLowerBound = lowerBound or self.targetTotalCostLowerBound
	
end

return OneVsAll