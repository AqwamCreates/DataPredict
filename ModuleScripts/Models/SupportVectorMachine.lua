local BaseModel = require(script.Parent.BaseModel)

SupportVectorMachineModel = {}

SupportVectorMachineModel.__index = SupportVectorMachineModel

setmetatable(SupportVectorMachineModel, BaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

local defaultMaxNumberOfIterations = 500

local defaultLearningRate = 0.3

local defaultCvalue = 0.3

local defaultDistanceFunction = "euclidean"

local defaultTargetCost = 0

local distanceFunctionList = {

	["manhattan"] = function (x1, x2)

		local part1 = AqwamMatrixLibrary:subtract(x1, x2)

		part1 = AqwamMatrixLibrary:applyFunction(math.abs, part1)

		local distance = AqwamMatrixLibrary:sum(part1)

		return distance 

	end,

	["euclidean"] = function (x1, x2)

		local part1 = AqwamMatrixLibrary:subtract(x1, x2)

		local part2 = AqwamMatrixLibrary:power(part1, 2)

		local part3 = AqwamMatrixLibrary:sum(part2)
		
		local distance = math.sqrt(part3)

		return distance 
		
	end,

}

local function calculateDistance(vector1, vector2, distanceFunction)

	return distanceFunctionList[distanceFunction](vector1, vector2) 

end

local function calculateCost(modelParameters, featureMatrix, labelVector, distanceFunction, cValue)
	
	local hypothesisVector = AqwamMatrixLibrary:dotProduct(featureMatrix, modelParameters)
	
	local distanceVector = calculateDistance(hypothesisVector, labelVector, distanceFunction)
	
	local squaredDistanceVector = AqwamMatrixLibrary:multiply(distanceVector, distanceVector)
	
	local sum = AqwamMatrixLibrary:sum(squaredDistanceVector)
	
	local cost = (1/2 * sum)

	return cost

end

local function gradientDescent(modelParameters, featureMatrix, labelVector, distanceFunction, cValue)

	local numberOfData = #featureMatrix

	local hypothesisVector = AqwamMatrixLibrary:dotProduct(featureMatrix, modelParameters)

	local calculatedError = AqwamMatrixLibrary:add(hypothesisVector, labelVector)
	
	local calculatedErrorWithFeatureMatrix = AqwamMatrixLibrary:multiply(calculatedError, featureMatrix)

	local calculatedSumError =  AqwamMatrixLibrary:verticalSum(calculatedError)

	local costFunctionDerivatives = AqwamMatrixLibrary:multiply((1/numberOfData), calculatedSumError)

	return costFunctionDerivatives

end

function SupportVectorMachineModel.new(maxNumberOfIterations, learningRate, cValue, distanceFunction, targetCost)
	
	local NewSupportVectorMachine = BaseModel.new()
	
	setmetatable(NewSupportVectorMachine, SupportVectorMachineModel)
	
	NewSupportVectorMachine.maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations

	NewSupportVectorMachine.learningRate = learningRate or defaultLearningRate
	
	NewSupportVectorMachine.cValue = cValue or defaultCvalue

	NewSupportVectorMachine.distanceFunction = distanceFunction or defaultDistanceFunction

	NewSupportVectorMachine.targetCost = targetCost or defaultTargetCost

	NewSupportVectorMachine.validationFeatureMatrix = nil

	NewSupportVectorMachine.validationLabelVector = nil
	
	NewSupportVectorMachine.Optimizer = nil
	
	return NewSupportVectorMachine

end

function SupportVectorMachineModel:setOptimizer(Optimizer)

	self.Optimizer = Optimizer

end

function SupportVectorMachineModel:setParameters(maxNumberOfIterations, learningRate, cValue, distanceFunction, targetCost)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate
	
	self.cValue = cValue or self.cValue

	self.distanceFunction = distanceFunction or self.distanceFunction

	self.targetCost = targetCost or self.targetCost

end

function SupportVectorMachineModel:setCValue(cValue)

	self.cValue = cValue or self.cValue

end

function SupportVectorMachineModel:train(featureMatrix, labelVector)
	
	local cost

	local costArray = {}

	local numberOfIterations = 0
	
	local costFunctionDerivatives
	
	local delta

	if (#featureMatrix ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows!") end

	if (self.ModelParameters) then

		if (#featureMatrix[1] ~= #self.ModelParameters) then error("The number of features are not the same as the model parameters!") end

	else

		self.ModelParameters = self:initializeMatrixBasedOnMode(#featureMatrix[1], 1)

	end	

	repeat

		numberOfIterations += 1
		
		costFunctionDerivatives = gradientDescent(self.ModelParameters, featureMatrix, labelVector, self.distanceFunction, self.learningRate, self.cValue)

		if (self.Optimizer) then 

			costFunctionDerivatives = self.Optimizer:calculate(costFunctionDerivatives, delta) 

		end

		delta = AqwamMatrixLibrary:multiply(self.learningRate, costFunctionDerivatives)

		self.ModelParameters = AqwamMatrixLibrary:add(self.ModelParameters, delta)

		cost = calculateCost(self.ModelParameters, featureMatrix, labelVector, self.distanceFunction, self.learningRate, self.cValue)
		
		table.insert(costArray, cost)
		
		self:printCostAndNumberOfIterations(cost, numberOfIterations)

	until (numberOfIterations == self.maxNumberOfIterations) or (math.abs(cost) <= self.targetCost)

	if (cost == math.huge) then warn("The model diverged! Please repeat the experiment again or change the argument values.") end
	
	if self.Optimizer then self.Optimizer:reset() end

	return costArray

end

function SupportVectorMachineModel:predict(featureMatrix)
	
	local hypothesis = AqwamMatrixLibrary:dotProduct(featureMatrix, self.ModelParameters)
	
	if (hypothesis > 0) then
		
		return 1
		
	elseif (hypothesis < 0) then
		
		return -1
		
	end 

end

return SupportVectorMachineModel
