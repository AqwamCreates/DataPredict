local BaseModel = require(script.Parent.BaseModel)

local AffinityPropagationModel = {}

AffinityPropagationModel.__index = AffinityPropagationModel

setmetatable(AffinityPropagationModel, BaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

local defaultMaxNumberOfIterations = 500

local defaultTargetCost = 0

local defaultDamping = 0.5a

local function initializePreferences(featureMatrix, numberOfData, numberOfFeatures)
	
	local preferences = {}
	
	for i = 1, numberOfData do
		
		local rowSum = 0
		
		for j = 1, numberOfFeatures do
			
			for k = 1, numberOfData do
				
				rowSum = rowSum + math.abs(featureMatrix[i][j] - featureMatrix[k][j])
				
			end
			
		end
		
		preferences[i] = -rowSum / (numberOfData * numberOfFeatures)
		
	end
	
	return preferences
	
end

local function computeSimilarities(similarities, featureMatrix, preferences, numberOfData, numberOfFeatures)
	
	for i = 1, numberOfData do
		
		for j = 1, numberOfData do
			
			local similarity = -math.huge
			
			for k = 1, numberOfFeatures do
				
				similarity = math.max(similarity, featureMatrix[i][k] * featureMatrix[j][k])
				
			end
			
			similarities[i][j] = similarity + preferences[i]
			
		end
		
	end
	
end

local function updateResponsibilities(responsibilities, similarities, availabilities, numberOfData)
	
	for i = 1, numberOfData do
		
		for j = 1, numberOfData do
			
			local maxResponsibility = -math.huge
			
			for k = 1, numberOfData do
				
				if k ~= j then
					
					maxResponsibility = math.max(maxResponsibility, similarities[i][k] + availabilities[k][j])
					
				end
				
			end
			
			responsibilities[i][j] = similarities[i][j] - maxResponsibility
			
		end
		
	end
	
end

local function updateAvailabilities(availabilities, responsibilities, numberOfData, damping)
	
	local maxAvailability

	for i = 1, numberOfData, 1 do

		for j = 1, numberOfData, 1 do

			if (i ~= j) then

				maxAvailability = -math.huge

				for k = 1, numberOfData, 1 do
					
					if (k ~= i) and (k ~= j) then
						
						maxAvailability = math.max(maxAvailability, 0, responsibilities[k][j])
						
					end
					
				end
				
				availabilities[i][j] = damping * (responsibilities[j][j] + maxAvailability) + (1 - damping) * availabilities[i][j]
				
			end
			
		end
		
	end
	
end

local function checkConvergence(clusters, responsibilities, availabilities, numberOfData)
	
	local converged = true
	
	for i = 1, numberOfData do
		
		local maxResponsibility = -math.huge
		
		local maxCluster = 0
		
		for j = 1, numberOfData do
			
			if ((responsibilities[i][j] + availabilities[i][j]) > maxResponsibility) then
				
				maxResponsibility = responsibilities[i][j] + availabilities[i][j]
				
				maxCluster = j
				
			end
			
		end
		
		if (clusters[i] ~= maxCluster) then
			
			converged = false
			
			clusters[i] = maxCluster
			
		end
		
	end
	
	return converged
	
end

local function calculateCost(ModelParameters, responsibilities)
	
	local totalCost = 0
	
	for i = 1, #ModelParameters do
		
		totalCost += responsibilities[i][ModelParameters[i]]
		
	end
	
	return totalCost
	
end

local function assignClusters(availabilities, responsibilities, numberOfData)
	
	local clusters = {}
	
	local maxResponsibility
	
	for i = 1, numberOfData do

		maxResponsibility = -math.huge

		for j = 1, numberOfData do

			if ((responsibilities[i][j] + availabilities[i][j]) > maxResponsibility) then

				clusters[i] = j

				maxResponsibility = responsibilities[i][j] + availabilities[i][j]

			end

		end

	end
	
	return clusters
	
end

function AffinityPropagationModel.new(maxNumberOfIterations, damping, targetCost)

	local NewAffinityPropagationModel = BaseModel.new()

	setmetatable(NewAffinityPropagationModel, AffinityPropagationModel)

	NewAffinityPropagationModel.maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations
	
	NewAffinityPropagationModel.damping = damping or defaultDamping
	
	NewAffinityPropagationModel.targetCost = targetCost or defaultTargetCost
	
	NewAffinityPropagationModel.appendPreviousFeatureMatrix = false
	
	NewAffinityPropagationModel.previousFeatureMatrix = nil

	return NewAffinityPropagationModel

end

function AffinityPropagationModel:setParameters(maxNumberOfIterations, damping, targetCost)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations
	
	self.damping = damping or self.damping
	
	self.targetCost = targetCost or self.targetCost

end

function AffinityPropagationModel:canAppendPreviousFeatureMatrix(option)
	
	self.appendPreviousFeatureMatrix = option
	
end

function AffinityPropagationModel:train(featureMatrix)
	
	if (self.previousFeatureMatrix) and (self.appendPreviousFeatureMatrix) then
		
		featureMatrix = AqwamMatrixLibrary:verticalConcatenate(featureMatrix, self.previousFeatureMatrix)
		
	end
	
	local numberOfData = #featureMatrix
	
	local numberOfFeatures = #featureMatrix[1]
	
	local preferences = initializePreferences(featureMatrix, numberOfData, numberOfFeatures)
	
	local similarities = AqwamMatrixLibrary:createMatrix(numberOfData, numberOfData)
	
	local responsibilities = AqwamMatrixLibrary:createMatrix(numberOfData, numberOfData)
	
	local availabilities = AqwamMatrixLibrary:createMatrix(numberOfData, numberOfData)
	
	local numberOfIterations = 0
	
	local clusters = {}
	
	local isConverged
	
	local cost
	
	repeat
		
		numberOfIterations += 1
		
		computeSimilarities(similarities, featureMatrix, preferences, numberOfData, numberOfFeatures)

		updateResponsibilities(responsibilities, similarities, availabilities, numberOfData)

		updateAvailabilities(availabilities, responsibilities, numberOfData, self.damping)
		
		isConverged = checkConvergence(clusters, responsibilities, availabilities, numberOfData)
		
		self.ModelParameters = assignClusters(availabilities, responsibilities, numberOfData)
		
		cost = calculateCost(self.ModelParameters, responsibilities)
		
		self:printCostAndNumberOfIterations(cost, numberOfIterations)
		
	until (self.maxNumberOfIterations) or (cost <= self.targetCost) or (isConverged)
	
	if (cost == math.huge) then warn("The model diverged! Please repeat the experiment again or change the argument values.") end
	
	self.previousFeatureMatrix = featureMatrix
	
end

function AffinityPropagationModel:predict(featureMatrix)
	
	if (self.previousFeatureMatrix == nil) then error("There are no feature matrix stored in this model. Please retrain the model.") end
	
	local similarity
	
	local similarities = {}
	
	local predictedCluster = 0
	
	local maxSimilarity = -math.huge

	for i = 1, #self.previousFeatureMatrix do
		
		similarity = 0
		
		for j = 1, #featureMatrix do
			
			similarity = similarity - (self.previousFeatureMatrix[i][j] - featureMatrix[1][j])^2
			
		end
		
		table.insert(similarities, similarity)
		
	end
	
	for i = 1, #self.previousFeatureMatrix do
		
		if (similarities[i] > maxSimilarity) then
			
			maxSimilarity = similarities[i]
			
			predictedCluster = self.ModelParameters[i]
			
		end
		
	end

	return predictedCluster, maxSimilarity
	
end

function AffinityPropagationModel:clearPreviousFeatureMatrix()
	
	self.previousFeatureMatrix = nil
	
end

return AffinityPropagationModel
