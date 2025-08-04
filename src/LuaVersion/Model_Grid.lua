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

local GradientMethodBaseModel = require("Model_GradientMethodBaseModel")

GridModel = {}

GridModel.__index = GridModel

setmetatable(GridModel, GradientMethodBaseModel)

local AqwamTensorLibrary = require("AqwamTensorLibrary")

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
	
	local ModelParameters = self.ModelParameters
	
	local outputMatrix = {}

	local modelParametersRowIndexTable = {}

	local numberOfClassesList = #self.ClassesList
	
	if (ModelParameters) then

		if (#featureMatrix[1] ~= #self.ModelParameters) then error("The number of features are not the same as the model parameters!") end

	else

		ModelParameters = self:initializeMatrixBasedOnMode(#featureMatrix[1], numberOfClassesList)

	end
	
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
	
	local ModelParameters = self.ModelParameters
	
	local modelParametersRowIndexTable =  self.modelParametersRowIndexTable
	
	local costDerivativeMatrix = AqwamMatrixLibrary:createMatrix(#ModelParameters, #ModelParameters[1])
	
	for lossMatrixIndex, rowVector in ipairs(lossMatrix) do
		
		local modelParametersRowIndex = modelParametersRowIndexTable[lossMatrixIndex]
		
		if (modelParametersRowIndex == 0) then continue end
		
		local costDerivativeVector = AqwamMatrixLibrary:add({costDerivativeMatrix[modelParametersRowIndex]}, {rowVector})
		
		costDerivativeMatrix[modelParametersRowIndex] = costDerivativeVector[modelParametersRowIndex]
		
	end
	
	self.ModelParameters = AqwamMatrixLibrary:subtract(ModelParameters, costDerivativeMatrix)
	
	if (clearTables) then
		
		self.modelParametersRowIndexTable = nil
		
	end
	
	if (self.areGradientsSaved) then self.Gradients = costDerivativeMatrix end
	
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
	
	local classesList = #self.ClassesList
	
	local numberOfData = #featureMatrix[1]
	
	if (#classesList == 0) then error("The classes list is empty!") end
	
	if (#featureMatrix ~= #labelMatrix) then error("The feature matrix and the label vector does not contain the same number of rows!") end
	
	AqwamMatrixLibrary:printMatrix(self.ModelParameters)
	
	outputMatrix = self:forwardPropagate(featureMatrix, true)
	
	lossMatrix = AqwamMatrixLibrary:subtract(outputMatrix, labelMatrix)
	
	costFunctionDerivatives = self:backPropagate(lossMatrix, true)
	
	return nil
	
end

function GridModel:predict(featureMatrix, returnOriginalOutput)
	
	local outputMatrix = self:forwardPropagate(featureMatrix, false)
	
	if (returnOriginalOutput == true) then return outputMatrix end
	
	local predictedLabelVector, highestValueVector = self:getLabelFromOutputMatrix(outputMatrix)

	return predictedLabelVector, highestValueVector

end

return GridModel
