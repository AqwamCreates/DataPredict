local AqwamMachineLearningModels = script.Parent:GetChildren()
local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

local ModelChecking = {}

function ModelChecking:testClassificationModel(MachineLearningModel, featureMatrix, labelVector) -- only works with supervised learning
	
	local modelOutputVector = AqwamMatrixLibrary:createMatrix(#labelVector, 1)
	
	local modelOutput
	
	local featureVector
	
	local accuracy
	
	local correctAtDataArray = {}
	
	local wrongAtDataArray = {}
	
	local numberOfData = #featureMatrix
	
	for data = 1, numberOfData, 1 do
		
		featureVector = {featureMatrix[1]}
		
		modelOutput = MachineLearningModel:predict(featureVector)
		
		modelOutputVector[data][1] = modelOutput
		
		if (modelOutput == labelVector[data][1]) then
			
			table.insert(correctAtDataArray, data)
			
		else
			
			table.insert(wrongAtDataArray, data)
			
		end
		
	end
	
	accuracy = #correctAtDataArray / numberOfData
	
	return accuracy, correctAtDataArray, wrongAtDataArray, modelOutputVector
	
end

function ModelChecking:testRegressionModel(MachineLearningModel, featureMatrix, labelVector)

	local modelOutputVector = AqwamMatrixLibrary:createMatrix(#labelVector, 1)

	local modelOutput

	local featureVector
	
	local errorVector
	
	local totalError
	
	local averageError

	local numberOfData = #featureMatrix
	
	for data = 1, numberOfData, 1 do

		featureVector = {featureMatrix[1]}

		modelOutput = MachineLearningModel:predict(featureVector)

		modelOutputVector[data][1] = modelOutput

	end
	
	errorVector = AqwamMatrixLibrary:subtract(modelOutputVector, labelVector)
	
	totalError = AqwamMatrixLibrary:verticalSum(errorVector)
	
	averageError = totalError/numberOfData
	
	return averageError, errorVector, modelOutputVector
	
end

return ModelChecking
