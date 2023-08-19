local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

OneVsAll = {}

OneVsAll.__index = OneVsAll

local defaultMaxNumberOfIterations = 500

local defaultTargetCost = 0

function OneVsAll.new(maxNumberOfIterations, useNegativeOneBinaryLabel, targetCost)
	
	local NewOneVsAll = {}
	
	setmetatable(NewOneVsAll, OneVsAll)
	
	NewOneVsAll.maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations
	
	NewOneVsAll.useNegativeOneBinaryLabel = useNegativeOneBinaryLabel or false
	
	NewOneVsAll.targetCost = defaultTargetCost
	
	NewOneVsAll.ModelsArray = nil
	
	NewOneVsAll.ClassesList = {}
	
	return NewOneVsAll
	
end

function OneVsAll:getBooleanOrDefaultOption(boolean, defaultBoolean)

	if (type(boolean) == "nil") then return defaultBoolean end

	return boolean

end

function OneVsAll:checkIfModelsSet()
	
	local typeOfModelsArray = typeof(self.ModelsArray)

	if (typeOfModelsArray ~= "table") then error("No models set!") end
	
end

function OneVsAll:setParameters(maxNumberOfIterations, useNegativeOneBinaryLabel, targetCost)
	
	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations
	
	self.useNegativeOneBinaryLabel = self:getBooleanOrDefaultOption(useNegativeOneBinaryLabel, self.useNegativeOneBinaryLabel)
	
	self.targetCost = targetCost or self.targetCost 
	
end

function OneVsAll:setModels(...)
	
	local inputtedModelsArray = {...}
	
	local proccesedModelsArray = ((#inputtedModelsArray > 0) and inputtedModelsArray) or nil
	
	if (proccesedModelsArray ~= nil) then
		
		for m, Model in ipairs(proccesedModelsArray) do Model:setPrintOutput(false) end
		
	end
	
	self.ModelsArray = proccesedModelsArray
	
end

function OneVsAll:setAllModelsParameters(...)
	
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

	local labelVectorColumn = AqwamMatrixLibrary:transpose(labelVector)

	for i, value in ipairs(labelVectorColumn[1]) do

		if table.find(classesList, value) then continue end

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
		
		print("Iteration: " .. numberOfIterations .. "\t\tCost: " .. totalCost)
		
	until (numberOfIterations >= self.maxNumberOfIterations) or (totalCost <= self.targetCost)
	
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

function OneVsAll:setModelParameters(...)
	
	self:checkIfModelsSet()
	
	local ModelParametersArray = {...}
	
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

return OneVsAll
