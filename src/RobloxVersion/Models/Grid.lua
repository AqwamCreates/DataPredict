local GradientMethodBaseModel = require(script.Parent.GradientMethodBaseModel)

GridModel = {}

GridModel.__index = GridModel

setmetatable(GridModel, GradientMethodBaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

function GridModel.new()
	
	local NewGridModel = GradientMethodBaseModel.new()
	
	setmetatable(NewGridModel, GridModel)
	
	NewGridModel.ClassesList = {}
	
	return NewGridModel
	
end

function GridModel:setClassesList(classesList)
	
	self.ClassesList = classesList
	
end

function GridModel:forwardPropagate(featureMatrix, saveTables)
	
	local outputMatrix = {}
	
	local modelParametersRowIndexTable = {}
	
	local numberOfClassesList = #self.ClassesList
	
	local ModelParameters = self.ModelParameters
	
	for i, rowVector in ipairs(featureMatrix) do
		
		local modelParametersRowIndex = table.find(rowVector, 1)
		
		if not modelParametersRowIndex then outputMatrix[i] = table.create(numberOfClassesList, 0) end
		
		table.insert(outputMatrix, ModelParameters[modelParametersRowIndex])
		
		if (not saveTables) then continue end
		
		table.insert(modelParametersRowIndexTable, modelParametersRowIndex or 0)
		
	end
	
	if (saveTables) then
		
		self.modelParametersRowIndexTable = modelParametersRowIndexTable
		
	end
	
	return outputMatrix
	
end

function GridModel:backPropagate(lossMatrix, clearTables)
	
	local modelParametersRowIndexTable =  self.modelParametersRowIndexTable
	
	local costDerivativeMatrix = AqwamMatrixLibrary:createMatrix(#lossMatrix, #lossMatrix[1])
	
	for i, rowVector in ipairs(lossMatrix) do
		
		local modelParametersRowIndex = modelParametersRowIndexTable[i]
		
		if (modelParametersRowIndex == 0) then continue end
		
	end
	
	if (clearTables) then
		
		self.modelParametersRowIndexTable = nil
		
	end
	
	return costDerivativeMatrix
	
end

function GridModel:getLabelFromOutputMatrix(outputMatrix)

	local numberOfNeuronsAtFinalLayer = self.numberOfNeuronsTable[#self.numberOfNeuronsTable]

	local predictedLabelVector = AqwamMatrixLibrary:createMatrix(#outputMatrix, 1)

	local highestValueVector = AqwamMatrixLibrary:createMatrix(#outputMatrix, 1)

	local highestValue

	local outputVector

	local classIndex

	local predictedLabel

	for i = 1, #outputMatrix, 1 do

		outputVector = {outputMatrix[i]}

		predictedLabel, highestValue = self:fetchHighestValueInVector(outputVector)

		predictedLabelVector[i][1] = predictedLabel

		highestValueVector[i][1] = highestValue

	end

	return predictedLabelVector, highestValueVector

end

function GridModel:train(featureMatrix, labelMatrix)
	
	local costFunctionDerivatives
	
	local predictedMatrix
	
	local outputMatrix
	
	local lossMatrix
	
	local numberOfData = #featureMatrix[1]
	
	if (#featureMatrix ~= #labelMatrix) then error("The feature matrix and the label vector does not contain the same number of rows!") end

	if (self.ModelParameters) then
		
		if (#featureMatrix[1] ~= #self.ModelParameters) then error("The number of features are not the same as the model parameters!") end
		
	else
		
		self.ModelParameters = self:initializeMatrixBasedOnMode(#featureMatrix[1], #self.ClassesList)
		
	end
	
	outputMatrix = self:forwardPropagate(featureMatrix, true)
	
	lossMatrix = AqwamMatrixLibrary:subtract(outputMatrix, labelMatrix)
	
	costFunctionDerivatives = self:backPropagate(lossMatrix, true)

	if (self.areGradientsSaved) then self.Gradients = costFunctionDerivatives end
	
	return nil
	
end

function GridModel:predict(featureMatrix, returnOriginalOutput)
	
	local outputMatrix = self:forwardPropagate(featureMatrix, true)
	
	if (returnOriginalOutput == true) then return outputMatrix end
	
	local predictedLabelVector, highestValueVector = self:getLabelFromOutputMatrix(outputMatrix)

	return predictedLabelVector, highestValueVector

end

return GridModel
