AdaptiveMomentEstimationOptimizer = {}

AdaptiveMomentEstimationOptimizer.__index = AdaptiveMomentEstimationOptimizer

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

local defaultBeta1 = 0.9

local defaultBeta2 = 0.999

function AdaptiveMomentEstimationOptimizer.new(Beta1, Beta2)

	local NewAdaptiveMomentEstimationOptimizer = {}

	setmetatable(NewAdaptiveMomentEstimationOptimizer, AdaptiveMomentEstimationOptimizer)

	NewAdaptiveMomentEstimationOptimizer.PreviousSumOfGradientSquaredMatrix = nil
	
	NewAdaptiveMomentEstimationOptimizer.Beta1 = Beta1 or defaultBeta1
	
	NewAdaptiveMomentEstimationOptimizer.Beta2 = Beta2 or defaultBeta2

	return NewAdaptiveMomentEstimationOptimizer

end

function AdaptiveMomentEstimationOptimizer:setBeta1(Beta1)
	
	self.Beta1 = Beta1
	
end

function AdaptiveMomentEstimationOptimizer:setBeta2(Beta2)
		
	self.Beta2 = Beta2
	
end


function AdaptiveMomentEstimationOptimizer:calculate(ModelParametersDerivatives)

	if (self.PreviousSumOfGradientSquaredMatrix == nil) then

		self.PreviousSumOfGradientSquaredMatrix = AqwamMatrixLibrary:createMatrix(#ModelParametersDerivatives, #ModelParametersDerivatives[1])

	end
	
	local MomentumPart1Matrix = AqwamMatrixLibrary:multiply(self.Beta1, self.PreviousSumOfGradientSquaredMatrix)
	
	local MomentumPart2Matrix = AqwamMatrixLibrary:multiply((1 - self.Beta1), ModelParametersDerivatives)
	
	local MomentumMatrix = AqwamMatrixLibrary:add(MomentumPart1Matrix, MomentumPart2Matrix)
	
	local RMSPropPart1Matrix = AqwamMatrixLibrary:multiply(self.Beta2, self.PreviousSumOfGradientSquaredMatrix)
	
	local SquaredSumOfGradientMatrix = AqwamMatrixLibrary:power(ModelParametersDerivatives, 2)
	
	local RMSPropPart2Matrix = AqwamMatrixLibrary:multiply((1 - self.Beta2), SquaredSumOfGradientMatrix)
	
	local RMSPropMatrix = AqwamMatrixLibrary:add(RMSPropPart1Matrix, RMSPropPart2Matrix)
	
	local AdamMatrix = AqwamMatrixLibrary:divide(MomentumMatrix, RMSPropMatrix)

	return AdamMatrix

end

function AdaptiveMomentEstimationOptimizer:reset()

	self.PreviousSumOfGradientSquaredMatrix = nil

end

return AdaptiveMomentEstimationOptimizer

