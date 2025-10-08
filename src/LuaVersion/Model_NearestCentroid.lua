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

local BaseModel = require("Model_BaseModel")

NearestCentroid = {}

NearestCentroid.__index = NearestCentroid

setmetatable(NearestCentroid, BaseModel)

local defaultDistanceFunction = "Euclidean"

local distanceFunctionList = {

	["Manhattan"] = function(x1, x2)

		local part1 = AqwamTensorLibrary:subtract(x1, x2)

		part1 = AqwamTensorLibrary:applyFunction(math.abs, part1)

		local distance = AqwamTensorLibrary:sum(part1)

		return distance 

	end,

	["Euclidean"] = function(x1, x2)

		local part1 = AqwamTensorLibrary:subtract(x1, x2)

		local part2 = AqwamTensorLibrary:power(part1, 2)

		local part3 = AqwamTensorLibrary:sum(part2)

		local distance = math.sqrt(part3)

		return distance 

	end,

	["Cosine"] = function(x1, x2)

		local dotProductedX = AqwamTensorLibrary:dotProduct(x1, AqwamTensorLibrary:transpose(x2))

		local x1MagnitudePart1 = AqwamTensorLibrary:power(x1, 2)

		local x1MagnitudePart2 = AqwamTensorLibrary:sum(x1MagnitudePart1)

		local x1Magnitude = math.sqrt(x1MagnitudePart2, 2)

		local x2MagnitudePart1 = AqwamTensorLibrary:power(x2, 2)

		local x2MagnitudePart2 = AqwamTensorLibrary:sum(x2MagnitudePart1)

		local x2Magnitude = math.sqrt(x2MagnitudePart2, 2)

		local normX = x1Magnitude * x2Magnitude

		local similarity = dotProductedX / normX

		local cosineDistance = 1 - similarity

		return cosineDistance

	end,

}

local function createDistanceMatrix(featureMatrix, storedFeatureMatrix, distanceFunction)

	local numberOfData = #featureMatrix

	local numberOfStoredData = #storedFeatureMatrix

	local distanceMatrix = AqwamTensorLibrary:createTensor({numberOfData, numberOfStoredData}, 0)

	local calculateDistance = distanceFunctionList[distanceFunction]

	for datasetIndex = 1, numberOfData, 1 do

		for storedDatasetIndex = 1, numberOfStoredData, 1 do

			distanceMatrix[datasetIndex][storedDatasetIndex] = calculateDistance({featureMatrix[datasetIndex]}, {storedFeatureMatrix[storedDatasetIndex]})

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

function NearestCentroid:processLabelVector(labelVector)

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

function NearestCentroid:convertLabelVectorToLogisticMatrix(labelVector)

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

function NearestCentroid:separateFeatureMatrixByClass(featureMatrix, labelMatrix)

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

function NearestCentroid.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewNearestCentroid = BaseModel.new(parameterDictionary)

	setmetatable(NewNearestCentroid, NearestCentroid)
	
	NewNearestCentroid:setName("NearestCentroid")

	NewNearestCentroid.distanceFunction = parameterDictionary.distanceFunction or defaultDistanceFunction
	
	NewNearestCentroid.ClassesList = parameterDictionary.ClassesList or {}

	return NewNearestCentroid

end

function NearestCentroid:train(featureMatrix, labelVector)

	if (#featureMatrix ~= #labelVector) then error("The number of data in feature matrix and the label vector are not the same.") end
	
	self:processLabelVector(labelVector)
	
	local labelMatrix
	
	if (#labelVector[1] == 1) then
		
		labelMatrix = self:convertLabelVectorToLogisticMatrix(labelVector)
		
	else
		
		labelMatrix = labelVector
		
	end

	local numberOfFeatures = #featureMatrix[1]
	
	local numberOfClasses = #self.ClassesList
	
	local ModelParameters = self.ModelParameters or {}
	
	local centroidMatrix = ModelParameters[1] or AqwamTensorLibrary:createTensor({numberOfClasses, numberOfFeatures}, 0)
	
	local numberOfDataPointVector = ModelParameters[2] or AqwamTensorLibrary:createTensor({numberOfClasses, 1}, 0)
	
	local sumMatrix = AqwamTensorLibrary:multiply(centroidMatrix, numberOfDataPointVector)
	
	local extractedFeatureMatrixTable = self:separateFeatureMatrixByClass(featureMatrix, labelMatrix)

	for clusterIndex, featureMatrix in ipairs(extractedFeatureMatrixTable) do
		
		if (featureMatrix) then
			
			local sumVector = {sumMatrix[clusterIndex]}

			local subSumVector = AqwamTensorLibrary:sum(featureMatrix, 1)

			sumVector = AqwamTensorLibrary:add(sumVector, subSumVector)

			sumMatrix[clusterIndex] = sumVector[1]

			numberOfDataPointVector[clusterIndex][1] = numberOfDataPointVector[clusterIndex][1] + #featureMatrix
			
		end
		
	end
	
	centroidMatrix = AqwamTensorLibrary:divide(sumMatrix, numberOfDataPointVector)

	self.ModelParameters = {centroidMatrix, numberOfDataPointVector}

end

function NearestCentroid:predict(featureMatrix, returnOriginalOutput)
	
	local ModelParameters = self.ModelParameters

	if (not ModelParameters) then error("No model parameters.") end

	local centroidMatrix = ModelParameters[1]

	local distanceMatrix = createDistanceMatrix(featureMatrix, centroidMatrix, self.distanceFunction)

	if (returnOriginalOutput) then return distanceMatrix end
	
	local ClassesList = self.ClassesList
	
	local numberOfData = #featureMatrix
	
	local numberOfClasses = #ClassesList

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

return NearestCentroid
