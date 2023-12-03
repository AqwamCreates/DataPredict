--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

	Version 1.18.0

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
	
	DO NOT SELL, RENT, DISTRIBUTE THIS LIBRARY
	
	DO NOT SELL, RENT, DISTRIBUTE MODIFIED VERSION OF THIS LIBRARY
	
	DO NOT CLAIM OWNERSHIP OF THIS LIBRARY
	
	GIVE CREDIT AND SOURCE WHEN USING THIS LIBRARY IF YOUR USAGE FALLS UNDER ONE OF THESE CATEGORIES:
	
		- USED AS A VIDEO OR ARTICLE CONTENT
		- USED AS RESEARCH AND EDUCATION CONTENT
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

local requiredMatrixLibraryVersion = 1.8

local AqwamMatrixLibrary = require(script.AqwamRobloxMatrixLibraryLinker.Value)

local Models = script.Models

local AqwamCustomModels = script.AqwamCustomModels

local Others = script.Others

local Optimizers = script.Optimizers

local ExperienceReplays = script.ExperienceReplays

local ModelsDictionary = {
	
	LinearRegression = require(Models.LinearRegression),
	
	LogisticRegression = require(Models.LogisticRegression),
	
	KMeans = require(Models.KMeans),
	
	SupportVectorMachine = require(Models.SupportVectorMachine),
	
	NaiveBayes = require(Models.NaiveBayes),
	
	ExpectationMaximization = require(Models.ExpectationMaximization),
	
	NeuralNetwork = require(Models.NeuralNetwork),
	
	ActorCritic = require(Models.ActorCritic),
	
	AdvantageActorCritic = require(Models.AdvantageActorCritic),
	
	AsynchronousAdvantageActorCritic = require(Models.AsynchronousAdvantageActorCritic),
	
	QLearningNeuralNetwork = require(Models.QLearningNeuralNetwork),
	
	DoubleQLearningNeuralNetworkV1 = require(Models.DoubleQLearningNeuralNetworkV1),
	
	DoubleQLearningNeuralNetworkV2 = require(Models.DoubleQLearningNeuralNetworkV2),
	
	ClippedDoubleQLearningNeuralNetwork = require(Models.ClippedDoubleQLearningNeuralNetwork),
	
	StateActionRewardStateActionNeuralNetwork = require(Models.StateActionRewardStateActionNeuralNetwork),
	
	DoubleStateActionRewardStateActionNeuralNetworkV1 = require(Models.DoubleStateActionRewardStateActionNeuralNetworkV1),
	
	DoubleStateActionRewardStateActionNeuralNetworkV2 = require(Models.DoubleStateActionRewardStateActionNeuralNetworkV2),
	
	ExpectedStateActionRewardStateActionNeuralNetwork = require(Models.ExpectedStateActionRewardStateActionNeuralNetwork),
	
	DoubleExpectedStateActionRewardStateActionNeuralNetworkV1 = require(Models.DoubleExpectedStateActionRewardStateActionNeuralNetworkV1),
	
	DoubleExpectedStateActionRewardStateActionNeuralNetworkV2 = require(Models.DoubleExpectedStateActionRewardStateActionNeuralNetworkV2),
	
	REINFORCENeuralNetwork = require(Models.REINFORCENeuralNetwork),

	RecurrentNeuralNetwork = require(Models.RecurrentNeuralNetwork),

	LongShortTermMemory = require(Models.LongShortTermMemory),
	
	KMedoids = require(Models.KMedoids),
	
	AffinityPropagation = require(Models.AffinityPropagation),
	
	AgglomerativeHierarchical = require(Models.AgglomerativeHierarchical),
	
	DensityBasedSpatialClusteringOfApplicationsWithNoise = require(Models.DensityBasedSpatialClusteringOfApplicationsWithNoise),
	
	MeanShift = require(Models.MeanShift),
	
	NormalLinearRegression = require(Models.NormalLinearRegression),
	
}

local AqwamCustomModelsDictionary = {
	
	ConfidenceQLearningNeuralNetwork = require(AqwamCustomModels.ConfidenceQLearningNeuralNetwork),
	
}

local OptimizersDictionary = {
	
	RootMeanSquarePropagation = require(Optimizers.RootMeanSquarePropagation),
	
	Momentum = require(Optimizers.Momentum),
	
	AdaptiveGradient = require(Optimizers.AdaptiveGradient),
	
	AdaptiveGradientDelta = require(Optimizers.AdaptiveGradientDelta),
	
	AdaptiveMomentEstimation = require(Optimizers.AdaptiveMomentEstimation),
	
	AdaptiveMomentEstimationMaximum = require(Optimizers.AdaptiveMomentEstimationMaximum),
	
	NesterovAcceleratedAdaptiveMomentEstimation = require(Optimizers.NesterovAcceleratedAdaptiveMomentEstimation),
	
	Gravity = require(Optimizers.Gravity),
	
}

local ExperienceReplaysDictionary = {
	
	UniformExperienceReplay = require(ExperienceReplays.UniformExperienceReplay),
	
	PrioritizedExperienceReplay = require(ExperienceReplays.PrioritizedExperienceReplay),
	
	NStepExperienceReplay = require(ExperienceReplays.NStepExperienceReplay),
	
}

local OthersDictionary = {
	
	ModelChecker = require(Others.ModelChecker),
	
	ModelParametersMerger = require(Others.ModelParametersMerger),
	
	GradientDescentModifier =  require(Others.GradientDescentModifier),
	
	Regularization = require(Others.Regularization),
	
	StringSplitter = require(Others.StringSplitter),
	
	OnlineLearning = require(Others.OnlineLearning),
	
	DistributedLearning = require(Others.DistributedLearning),
	
	Tokenizer = require(Others.Tokenizer),
	
	OneVsAll = require(Others.OneVsAll),
	
}

local AqwamRobloxMachineLearningLibrary = {}

AqwamRobloxMachineLearningLibrary.Models =  ModelsDictionary

AqwamRobloxMachineLearningLibrary.AqwamCustomModels = AqwamCustomModelsDictionary

AqwamRobloxMachineLearningLibrary.Optimizers = OptimizersDictionary

AqwamRobloxMachineLearningLibrary.ExperienceReplays = ExperienceReplaysDictionary

AqwamRobloxMachineLearningLibrary.Others = OthersDictionary

local function checkVersion()
	
	local matrixLibraryVersion
	
	if (AqwamMatrixLibrary == nil) then error("\n\nMatrixL (or Aqwam's Matrix Library) is not linked to this library. \nPlease read the \"Installation & Usage \" in DataPredict's documentation for installation details. ") end
	
	local success = pcall(function()
		
		matrixLibraryVersion = AqwamMatrixLibrary:getVersion()
		
	end)
	
	if not success then matrixLibraryVersion = -1 end
	
	if (matrixLibraryVersion < requiredMatrixLibraryVersion) then warn("The matrix library is out-of-date. You may encounter some problems.") end
	
end

checkVersion()

return AqwamRobloxMachineLearningLibrary
