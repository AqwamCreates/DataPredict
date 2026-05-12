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

local BaseSolver = require(script.Parent.BaseSolver)

local IterativelyReweightedSolver = {}

IterativelyReweightedSolver.__index = IterativelyReweightedSolver

setmetatable(IterativelyReweightedSolver, BaseSolver)

function IterativelyReweightedSolver.new(parameterDictionary)
	
	local NewIterativelyReweightedSolver = BaseSolver.new(parameterDictionary)
	
	setmetatable(NewIterativelyReweightedSolver, IterativelyReweightedSolver)
	
	NewIterativelyReweightedSolver:setName("IterativelyReweighted")
	
	NewIterativelyReweightedSolver:setCalculateFunction(function(weightMatrix, inputMatrix, firstDerivativeMatrix, firstDerivativeLossMatrix)
		
		-- Can only cache from linear models since the derivative is a feature matrix. Hence, these values are constant.
		
		local isLinear = NewIterativelyReweightedSolver.isLinear
		
		local transposedJacobianMatrix = (isLinear and NewIterativelyReweightedSolver.cache)
		
		local numberOfdata = AqwamTensorLibrary:getDimensionSizeArray(firstDerivativeLossMatrix)[1]
		
		local jacobianMatrix = inputMatrix

		if (not isLinear) then jacobianMatrix = AqwamTensorLibrary:multiply(jacobianMatrix, firstDerivativeMatrix) end

		if (not transposedJacobianMatrix) then

			transposedJacobianMatrix = AqwamTensorLibrary:transpose(jacobianMatrix)
			
			if (isLinear) then NewIterativelyReweightedSolver.cache = transposedJacobianMatrix end

		end
		
		local weightMatrixDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(weightMatrix) 
		
		local firstDerivativeLossMatrixDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(inputMatrix)
		
		local numberOfData = firstDerivativeLossMatrixDimensionSizeArray[1]
		
		local numberOfFeatures = weightMatrixDimensionSizeArray[1]
		
		local numberOfOutputs = weightMatrixDimensionSizeArray[2]
		
		local firstDerivativeLossVector
		
		local pseudoInverseMatrix
		
		local weightChangeVector
		
		local weightChangeMatrix
		
		local diagonalMatrix
		
		pseudoInverseMatrix = transposedJacobianMatrix
		
		if (not isLinear) then
			
			diagonalMatrix = AqwamTensorLibrary:createTensor({numberOfData, numberOfData}, 0)
			
			for dataIndex, unwrappedFirstDerivativeVector in ipairs(firstDerivativeMatrix) do diagonalMatrix[dataIndex][dataIndex] = unwrappedFirstDerivativeVector[1] end
			
			pseudoInverseMatrix = AqwamTensorLibrary:dotProduct(pseudoInverseMatrix, diagonalMatrix)
			
		end
		
		pseudoInverseMatrix = AqwamTensorLibrary:dotProduct(pseudoInverseMatrix, jacobianMatrix)
		
		pseudoInverseMatrix = AqwamTensorLibrary:inverse(pseudoInverseMatrix)
		
		-- If it is non-invertible, then do not return any weight change values as it is likely to be a local minimum.
		
		if (not pseudoInverseMatrix) then return AqwamTensorLibrary:createTensor(weightMatrixDimensionSizeArray, 0) end
		
		pseudoInverseMatrix = AqwamTensorLibrary:dotProduct(pseudoInverseMatrix, transposedJacobianMatrix)
		
		local weightChangeBaseVector = pseudoInverseMatrix
		
		if (not isLinear) then weightChangeBaseVector = AqwamTensorLibrary:dotProduct(weightChangeBaseVector, diagonalMatrix) end
		
		for outputIndex = 1, numberOfOutputs, 1 do
			
			firstDerivativeLossVector = AqwamTensorLibrary:extract(firstDerivativeLossMatrix, {1, outputIndex}, {numberOfData, outputIndex})
			
			weightChangeVector = AqwamTensorLibrary:dotProduct(weightChangeBaseVector, firstDerivativeLossVector)
			
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
