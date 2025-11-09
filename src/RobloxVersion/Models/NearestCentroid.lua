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

local BaseModel = require(script.Parent.BaseModel)

local distanceFunctionDictionary = require(script.Parent.Parent.Cores.DistanceFunctionDictionary)

NearestCentroidModel = {}

NearestCentroidModel.__index = NearestCentroidModel

setmetatable(NearestCentroidModel, BaseModel)

local defaultDistanceFunction = "Euclidean"

local defaultMaximumNumberOfDataPoints = nil

local function createDistanceMatrix(distanceFunction, featureMatrix, centroidMatrix)

	local numberOfData = #featureMatrix

	local numberOfStoredData = #centroidMatrix

	local distanceMatrix = AqwamTensorLibrary:createTensor({numberOfData, numberOfStoredData}, 0)

	local calculateDistance = distanceFunctionDictionary[distanceFunction]

	for datasetIndex = 1, numberOfData, 1 do

		for storedDatasetIndex = 1, numberOfStoredData, 1 do

			distanceMatrix[datasetIndex][storedDatasetIndex] = calculateDistance({featureMatrix[datasetIndex]}, {centroidMatrix[storedDatasetIndex]})

		end

	end

	return distanceMatrix

end

local function areNumbersOnlyInList(list)

	for i, value in ipairs(list) do

		if (typeof(value) ~= "number") then return false end

	end

	return true

end

local function extractFeatureMatrixFromPosition(featureMatrix, positionList)

	local extractedFeatureMatrix = {}

	for i = 1, #featureMatrix, 1 do

		if table.find(positionList, i) then

			table.insert(extractedFeatureMatrix, featureMatrix[i])

		end	

	end

	return extractedFeatureMatrix

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

local function checkIfAnyLabelVectorIsNotRecognized(labelVector, ClassesList)

	for i = 1, #labelVector, 1 do

		if (not table.find(ClassesList, labelVector[i][1])) then return true end

	end

	return false

end

function NearestCentroidModel:processLabelVector(labelVector)

	local ClassesList = self.ClassesList

	if (#ClassesList == 0) then

		ClassesList = createClassesList(labelVector)

		local areNumbersOnly = areNumbersOnlyInList(ClassesList)

		if (areNumbersOnly) then table.sort(ClassesList, function(a,b) return a < b end) end

		self.ClassesList = ClassesList

	else

		if checkIfAnyLabelVectorIsNotRecognized(labelVector, ClassesList) then error("A value does not exist in the model\'s classes list is present in the label vector.") end

	end

end

function NearestCentroidModel:convertLabelVectorToLogisticMatrix(labelVector)

	if (typeof(labelVector) == "number") then

		labelVector = {{labelVector}}

	end

	local incorrectLabelValue

	local numberOfData = #labelVector

	local ClassesList = self.ClassesList

	local logisticMatrix = AqwamTensorLibrary:createTensor({numberOfData, #ClassesList}, 0)

	local label

	local labelPosition

	for data = 1, numberOfData, 1 do

		label = labelVector[data][1]

		labelPosition = table.find(ClassesList, label)

		if (labelPosition) then

			logisticMatrix[data][labelPosition] = 1

		end

	end

	return logisticMatrix

end

function NearestCentroidModel:separateFeatureMatrixByClass(featureMatrix, labelMatrix)

	local ClassesList = self.ClassesList

	local extractedFeatureMatrixTable = {}

	for i = 1, #ClassesList, 1 do extractedFeatureMatrixTable[i] = {} end

	for i, labelTable in ipairs(labelMatrix) do

		for j, labelValue in ipairs(labelTable) do

			if (labelValue > 0) then table.insert(extractedFeatureMatrixTable[j], featureMatrix[i]) end

		end

	end

	return extractedFeatureMatrixTable

end

function NearestCentroidModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewNearestCentroidModel = BaseModel.new(parameterDictionary)

	setmetatable(NewNearestCentroidModel, NearestCentroidModel)
	
	NewNearestCentroidModel:setName("NearestCentroid")

	NewNearestCentroidModel.distanceFunction = parameterDictionary.distanceFunction or defaultDistanceFunction
	
	NewNearestCentroidModel.ClassesList = parameterDictionary.ClassesList or {}
	
	NewNearestCentroidModel.maximumNumberOfDataPoints = BaseModel:getValueOrDefaultValue(parameterDictionary.maximumNumberOfDataPoints, defaultMaximumNumberOfDataPoints)

	return NewNearestCentroidModel

end

function NearestCentroidModel:train(featureMatrix, labelVector)
	
	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end
	
	local numberOfData = #featureMatrix

	if (numberOfData ~= #labelVector) then error("The number of data in feature matrix and the label vector are not the same.") end
	
	self:processLabelVector(labelVector)
	
	local labelMatrix
	
	if (#labelVector[1] == 1) then
		
		labelMatrix = self:convertLabelVectorToLogisticMatrix(labelVector)
		
	else
		
		labelMatrix = labelVector
		
	end

	local numberOfFeatures = #featureMatrix[1]
	
	local distanceFunction = self.distanceFunction
	
	local numberOfClasses = #self.ClassesList
	
	local maximumNumberOfDataPoints = self.maximumNumberOfDataPoints
	
	local ModelParameters = self.ModelParameters or {}
	
	local centroidMatrix = ModelParameters[1] or AqwamTensorLibrary:createTensor({numberOfClasses, numberOfFeatures}, 0)
	
	local numberOfDataPointVector = ModelParameters[2] or AqwamTensorLibrary:createTensor({numberOfClasses, 1}, 0)
	
	local sumMatrix = AqwamTensorLibrary:multiply(centroidMatrix, numberOfDataPointVector)
	
	local extractedFeatureMatrixTable = self:separateFeatureMatrixByClass(featureMatrix, labelMatrix)
	
	local numberOfDataPoints

	for clusterIndex, featureMatrix in ipairs(extractedFeatureMatrixTable) do
		
		if (featureMatrix) then
			
			local sumVector = {sumMatrix[clusterIndex]}

			local subSumVector = AqwamTensorLibrary:sum(featureMatrix, 1)

			sumVector = AqwamTensorLibrary:add(sumVector, subSumVector)

			sumMatrix[clusterIndex] = sumVector[1]
			
			numberOfDataPoints = numberOfDataPointVector[clusterIndex][1] + #featureMatrix
			
			if (type(maximumNumberOfDataPoints) == "number") then
				
				if (numberOfDataPoints > maximumNumberOfDataPoints) then numberOfDataPoints = 1 end
				
			end

			numberOfDataPointVector[clusterIndex][1] = numberOfDataPoints
			
		end
		
	end
	
	centroidMatrix = AqwamTensorLibrary:divide(sumMatrix, numberOfDataPointVector)
	
	local distanceMatrix = createDistanceMatrix(distanceFunction, featureMatrix, centroidMatrix)
	
	local costMatrix = AqwamTensorLibrary:multiply(distanceMatrix, labelMatrix)
	
	local cost = AqwamTensorLibrary:sum(costMatrix)
	
	cost = cost / numberOfData
	
	if (self.isOutputPrinted) then

		if (cost == math.huge) then warn("The model diverged.") end

		if (cost ~= cost) then warn("The model produced nan (not a number) values.") end

	end

	self.ModelParameters = {centroidMatrix, numberOfDataPointVector}
	
	return {cost}

end

function NearestCentroidModel:predict(featureMatrix, returnOriginalOutput)
	
	local ModelParameters = self.ModelParameters
	
	local ClassesList = self.ClassesList

	if (not ModelParameters) then
		
		local numberOfData = #featureMatrix
		
		local numberOfClasses = #ClassesList
		
		if (returnOriginalOutput) then return AqwamTensorLibrary:createTensor({numberOfData, numberOfClasses}, math.huge) end
		
		local dimensionSizeArray = {numberOfData, 1}
		
		local placeHolderLabelVector = AqwamTensorLibrary:createTensor(dimensionSizeArray, nil)
		
		local placeHolderLabelDistanceVector = AqwamTensorLibrary:createTensor(dimensionSizeArray, math.huge)
		
		return placeHolderLabelVector, placeHolderLabelDistanceVector
		
	end

	local centroidMatrix = ModelParameters[1]

	local distanceMatrix = createDistanceMatrix(self.distanceFunction, featureMatrix, centroidMatrix)

	if (returnOriginalOutput) then return distanceMatrix end

	local predictedLabelVector = {}
	
	local distanceVector = {}
	
	for dataIndex, unwrappedDistanceVector in ipairs(distanceMatrix) do
		
		local minimumDistance = math.huge

		local nearestClassIndex
		
		for classIndex, distance in ipairs(unwrappedDistanceVector) do
			
			if (distance < minimumDistance) then

				minimumDistance = distance

				nearestClassIndex = classIndex

			end			
			
		end
		
		predictedLabelVector[dataIndex] = {ClassesList[nearestClassIndex]}

		distanceVector[dataIndex] = {minimumDistance}
		
	end

	return predictedLabelVector, distanceVector

end

return NearestCentroidModel
