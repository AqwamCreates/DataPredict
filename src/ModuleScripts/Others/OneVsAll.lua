local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

OneVsAll = {}

OneVsAll.__index = OneVsAll

local defaultMaxNumberOfIterations = 500

function OneVsAll.new(maxNumberOfIterations)
	
	local NewOneVsAll = {}
	
	setmetatable(NewOneVsAll, OneVsAll)
	
	NewOneVsAll.maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations
	
	NewOneVsAll.ModelsArray = nil
	
	NewOneVsAll.ClassesList = {}
	
	return NewOneVsAll
	
end

function OneVsAll:checkIfModelsSet()
	
	local typeOfModelsArray = typeof(self.ModelsArray)

	if (typeOfModelsArray ~= "table") then error("No models set!") end
	
end

function OneVsAll:setParameters(maxNumberOfIterations)
	
	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations
	
end

function OneVsAll:setModels(...)
	
	local inputtedModels = {...}
	
	local proccesedModelsArray = ((#inputtedModels > 0) and inputtedModels) or nil
	
	self.ModelsArray = proccesedModelsArray
	
end

function OneVsAll:setAllModelsParameters(...)
	
	self:checkIfModelsSet()
	
	for _, Model in ipairs(self.ModelsArray) do Model:setParameters(...) end
	
end

function OneVsAll:setClassesList(classesList)

	self.ClassesList = classesList

end

function OneVsAll:train(featureMatrix, labelVector)
	
	self:checkIfModelsSet()
	
	local numberOfModels = #self.ModelsArray
	
	local costArray = {}
	
	local numberOfIterations = 0
	
	local modelCostArray
	
	repeat
		
		local totalCost = 0
		
		for _, Model in ipairs(self.ModelsArray) do

			modelCostArray = Model:train(featureMatrix, labelVector)

			totalCost += modelCostArray[#modelCostArray]

		end
		
		numberOfIterations += 1
		
		table.insert(costArray, totalCost)
		
		print("Iteration: " .. numberOfIterations .. "\t\tCost: " .. totalCost)
		
	until (numberOfIterations >= self.maxNumberOfIterations)
	
	return costArray
	
end

function OneVsAll:getBestPrediction(featureVector)
	
	local selectedModelNumber = 0
	
	local highestValue = -math.huge
	
	for m, Model in ipairs(self.ModelsArray) do 

		local allOutputVector = Model:predict(featureVector, true)

		local _, maximumValueIndex = AqwamMatrixLibrary:findMaximumValueInMatrix(allOutputVector)

		if (maximumValueIndex == nil) then continue end

		local value = maximumValueIndex[2]

		if (value <= highestValue) then continue end

		highestValue = value

		selectedModelNumber = m

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

return OneVsAll
