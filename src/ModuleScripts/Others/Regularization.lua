local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

Regularization = {}

Regularization.__index = Regularization

local defaultRegularizationMode = "L2"

local defaultLambda = 0.01

local function getBooleanOrDefaultOption(boolean, defaultBoolean)
	
	if (type(boolean) == "nil") then return defaultBoolean end

	return boolean
	
end

function Regularization.new(lambda, regularizationMode, hasBias)
	
	local NewRegularization = {}
	
	setmetatable(NewRegularization, Regularization)
	
	NewRegularization.lambda = lambda or defaultLambda
	
	NewRegularization.regularizationMode = regularizationMode or defaultRegularizationMode
	
	NewRegularization.hasBias = getBooleanOrDefaultOption(hasBias, true)
	
	return NewRegularization
	
end

function Regularization:setParameters(lambda, regularizationMode, hasBias)
	
	self.lambda = lambda or self.lambda
	
	self.regularizationMode = regularizationMode or self.regularizationMode
	
	self.hasBias = getBooleanOrDefaultOption(hasBias, self.hasBias)
	
end

function Regularization:getLambda()
	
	return self.lambda
	
end

function Regularization:calculateRegularizationDerivatives(ModelParameters, numberOfData)
	
	local ModelParametersSign
	
	local RegularizationDerivative
	
	if (self.regularizationMode == "L1") or (self.regularizationMode == "Lasso") then
		
		ModelParametersSign = AqwamMatrixLibrary:applyFunction(math.sign, ModelParameters)
		
		RegularizationDerivative = AqwamMatrixLibrary:multiply(ModelParametersSign, self.lambda, ModelParameters)
		
		RegularizationDerivative = AqwamMatrixLibrary:divide(RegularizationDerivative, numberOfData)
	
	elseif (self.regularizationMode == "L2") or (self.regularizationMode == "Ridge") then
		
		RegularizationDerivative = AqwamMatrixLibrary:multiply(2, self.lambda, ModelParameters)
		
		RegularizationDerivative = AqwamMatrixLibrary:divide(RegularizationDerivative, numberOfData)
		
	elseif (self.regularizationMode == "L1+L2") or (self.regularizationMode == "ElasticNet") then
		
		ModelParametersSign = AqwamMatrixLibrary:applyFunction(math.sign, ModelParameters)
		
		local RegularizationDerivativePart1 = AqwamMatrixLibrary:multiply(self.lambda, ModelParametersSign)
		
		local RegularizationDerivativePart2 = AqwamMatrixLibrary:multiply(self.lambda, ModelParameters)
		
		RegularizationDerivative = AqwamMatrixLibrary:add(RegularizationDerivativePart1, RegularizationDerivativePart2)
		
		RegularizationDerivative = AqwamMatrixLibrary:divide(RegularizationDerivative, numberOfData)

	else

		error("Regularization Mode Does Not Exist!")

	end
	
	return RegularizationDerivative
	
end

function Regularization:calculateRegularization(ModelParameters, numberOfData)
	
	local SquaredModelParameters 
	
	local AbsoluteModelParameters
	
	local SumSquaredModelParameters
	
	local SumAbsoluteModelParameters
	
	local regularizationValue
	
	if (self.regularizationMode == "L1") or (self.regularizationMode == "Lasso") then
		
		AbsoluteModelParameters = AqwamMatrixLibrary:applyFunction(math.abs, ModelParameters)
		
		SumAbsoluteModelParameters = AqwamMatrixLibrary:sum(AbsoluteModelParameters)
		
		regularizationValue = self.lambda * SumAbsoluteModelParameters
		
		regularizationValue = regularizationValue / numberOfData
		
	elseif (self.regularizationMode == "L2") or (self.regularizationMode == "Ridge") then
		
		SquaredModelParameters = AqwamMatrixLibrary:power(ModelParameters, 2)
		
		SumSquaredModelParameters = AqwamMatrixLibrary:sum(SquaredModelParameters)
		
		regularizationValue = self.lambda * SumSquaredModelParameters
		
		regularizationValue = regularizationValue / numberOfData
		
	elseif (self.regularizationMode == "L1+L2") or (self.regularizationMode == "ElasticNet") then
		
		SquaredModelParameters = AqwamMatrixLibrary:power(ModelParameters, 2)
		
		SumSquaredModelParameters = AqwamMatrixLibrary:sum(SquaredModelParameters)
		
		AbsoluteModelParameters = AqwamMatrixLibrary:applyFunction(math.abs, ModelParameters)
		
		SumAbsoluteModelParameters = AqwamMatrixLibrary:sum(AbsoluteModelParameters)
		
		local regularizationValuePart1 = self.lambda * SumSquaredModelParameters
		
		local regularizationValuePart2 = self.lambda * SumAbsoluteModelParameters
		
		regularizationValue = regularizationValuePart1 + regularizationValuePart2
		
		regularizationValue = regularizationValue / numberOfData
		
	else
		
		error("Regularization Mode Does Not Exist!")
		
	end
	
	return regularizationValue
	
end

return Regularization
