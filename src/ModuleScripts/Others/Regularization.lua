local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

Regularization = {}

Regularization.__index = Regularization

local defaultRegularizationMode = "L2"

local defaultLambda = 0.01

local function getBooleanOrDefaultOption(boolean, defaultBoolean)
	
	if (type(boolean) == "nil") then return defaultBoolean end

	return boolean
	
end

local function makeLambdaAtBiasZero(ModelParameters)
	
	for i = 1, #ModelParameters[1], 1 do
		
		ModelParameters[1][i] = 0
		
	end
	
	return ModelParameters
	
end

function Regularization.new(lambda, regularizationMode, hasBias)
	
	local NewRegularization = {}
	
	setmetatable(NewRegularization, Regularization)
	
	NewRegularization.lambda = lambda or defaultLambda
	
	NewRegularization.regularizationMode = regularizationMode or defaultRegularizationMode
	
	NewRegularization.hasBias = getBooleanOrDefaultOption(hasBias, false)
	
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
		
		if (self.hasBias) then RegularizationDerivative = makeLambdaAtBiasZero(RegularizationDerivative) end
	
	elseif (self.regularizationMode == "L2") or (self.regularizationMode == "Ridge") then
		
		RegularizationDerivative = AqwamMatrixLibrary:multiply(2, self.lambda, ModelParameters)
		
		if (self.hasBias) then RegularizationDerivative = makeLambdaAtBiasZero(RegularizationDerivative) end
		
	elseif (self.regularizationMode == "L1+L2") or (self.regularizationMode == "ElasticNet") then
		
		ModelParametersSign = AqwamMatrixLibrary:applyFunction(math.sign, ModelParameters)
		
		local RegularizationDerivativePart1 = AqwamMatrixLibrary:multiply(self.lambda, ModelParametersSign)
		
		local RegularizationDerivativePart2 = AqwamMatrixLibrary:multiply(self.lambda, ModelParameters)
		
		RegularizationDerivative = AqwamMatrixLibrary:add(RegularizationDerivativePart1, RegularizationDerivativePart2)
		
		if (self.hasBias) then RegularizationDerivative = makeLambdaAtBiasZero(RegularizationDerivative) end

	else

		error("Regularization Mode Does Not Exist!")

	end
	
	RegularizationDerivative = AqwamMatrixLibrary:divide(RegularizationDerivative, numberOfData)
	
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
		
		if (self.hasBias) then AbsoluteModelParameters = makeLambdaAtBiasZero(AbsoluteModelParameters) end
		
		SumAbsoluteModelParameters = AqwamMatrixLibrary:sum(AbsoluteModelParameters)
		
		regularizationValue = self.lambda * SumAbsoluteModelParameters
		
	elseif (self.regularizationMode == "L2") or (self.regularizationMode == "Ridge") then
		
		SquaredModelParameters = AqwamMatrixLibrary:power(ModelParameters, 2)
		
		if (self.hasBias) then SquaredModelParameters = makeLambdaAtBiasZero(SquaredModelParameters) end
		
		SumSquaredModelParameters = AqwamMatrixLibrary:sum(SquaredModelParameters)
		
		regularizationValue = self.lambda * SumSquaredModelParameters
		
	elseif (self.regularizationMode == "L1+L2") or (self.regularizationMode == "ElasticNet") then
		
		SquaredModelParameters = AqwamMatrixLibrary:power(ModelParameters, 2)
		
		if (self.hasBias) then SquaredModelParameters = makeLambdaAtBiasZero(SquaredModelParameters) end
		
		SumSquaredModelParameters = AqwamMatrixLibrary:sum(SquaredModelParameters)
		
		AbsoluteModelParameters = AqwamMatrixLibrary:applyFunction(math.abs, ModelParameters)
		
		SumAbsoluteModelParameters = AqwamMatrixLibrary:sum(AbsoluteModelParameters)
		
		local regularizationValuePart1 = self.lambda * SumSquaredModelParameters
		
		local regularizationValuePart2 = self.lambda * SumAbsoluteModelParameters
		
		regularizationValue = regularizationValuePart1 + regularizationValuePart2
		
	else
		
		error("Regularization Mode Does Not Exist!")
		
	end
	
	regularizationValue = regularizationValue / numberOfData
	
	return regularizationValue
	
end

return Regularization
