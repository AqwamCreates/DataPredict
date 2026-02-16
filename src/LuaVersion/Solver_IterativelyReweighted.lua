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

local BaseSolver = require("Solver_BaseSolver")

local IterativelyReweightedSolver = {}

IterativelyReweightedSolver.__index = IterativelyReweightedSolver

setmetatable(IterativelyReweightedSolver, BaseSolver)

function IterativelyReweightedSolver.new(parameterDictionary)
	
	local NewIterativelyReweightedSolver = BaseSolver.new(parameterDictionary)
	
	setmetatable(NewIterativelyReweightedSolver, IterativelyReweightedSolver)
	
	NewIterativelyReweightedSolver:setName("IterativelyReweighted")
	
	NewIterativelyReweightedSolver:setCalculateFunction(function(weightMatrix, firstDerivativeMatrix, firstDerivativeLossMatrix)
		
		-- Can only cache from linear models since the derivative is a feature matrix. Hence, these values are constant.
		
		local isLinear = NewIterativelyReweightedSolver.isLinear
		
		local transposedFirstDerivativeMatrix = (isLinear and NewIterativelyReweightedSolver.cache)
		
		local numberOfdata = AqwamTensorLibrary:getDimensionSize(firstDerivativeLossMatrix)[1]
		
		local diagonalMatrix = AqwamTensorLibrary:createTensor({numberOfdata, numberOfdata})

		if (not transposedFirstDerivativeMatrix) then

			transposedFirstDerivativeMatrix = AqwamTensorLibrary:transpose(firstDerivativeMatrix)
			
			if (isLinear) then NewIterativelyReweightedSolver.cache = transposedFirstDerivativeMatrix end

		end
		
		local weightMatrixDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(weightMatrix) 
		
		local firstDerivativeLossMatrixDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(firstDerivativeLossMatrix)
		
		local numberOfData = firstDerivativeLossMatrixDimensionSizeArray[1]
		
		local numberOfFeatures = weightMatrixDimensionSizeArray[1]
		
		local numberOfOutputs = weightMatrixDimensionSizeArray[2]
		
		local weightVectorDimensionSizeArray = {numberOfFeatures, 1}
		
		local firstDerivativeLossVector
		
		local pMatrix
		
		local weightChangeVector
		
		local weightChangeMatrix
		
		for outputIndex = 1, numberOfOutputs, 1 do
			
			firstDerivativeLossVector = AqwamTensorLibrary:extract(firstDerivativeLossMatrix, {1, outputIndex}, {numberOfData, outputIndex})
			
			for dataIndex, unwrappedErrorVector in ipairs(firstDerivativeLossVector) do diagonalMatrix[dataIndex][dataIndex] = unwrappedErrorVector[1] end
			
			pMatrix = AqwamTensorLibrary:dotProduct(transposedFirstDerivativeMatrix, diagonalMatrix, firstDerivativeMatrix)
			
			pMatrix = AqwamTensorLibrary:inverse(pMatrix)
			
			-- If it is non-invertible, then do not return any weight change values as it is likely to be a local minimum.
			
			if (pMatrix) then
				
				weightChangeVector = AqwamTensorLibrary:dotProduct(pMatrix, diagonalMatrix, firstDerivativeLossVector)
				
			else
				
				weightChangeVector = AqwamTensorLibrary:createTensor(weightVectorDimensionSizeArray, 0)
				
			end
			
			if (weightChangeMatrix) then
				
				weightChangeMatrix = AqwamTensorLibrary:concatenate(weightChangeMatrix, weightChangeVector, 2)
				
			else
				
				weightChangeMatrix = weightChangeVector
				
			end
			
		end

		return weightChangeMatrix
		
	end)
	
	return NewIterativelyReweightedSolver
	
end

return IterativelyReweightedSolver
