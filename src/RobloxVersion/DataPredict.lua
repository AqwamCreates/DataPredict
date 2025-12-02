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

local AqwamMachineDeepAndReinforcementLearningLibrary = {}

local Models = script.Models

local Regularizers = script.Regularizers

local Optimizers = script.Optimizers

local ValueSchedulers = script.ValueSchedulers

local GradientClippers = script.GradientClippers

local ExperienceReplays = script.ExperienceReplays

local QuickSetups = script.QuickSetups

local EligibilityTraces = script.EligibilityTraces

local ReinforcementLearningStrategies = script.ReinforcementLearningStrategies

local DistributedTrainingStrategies = script.DistributedTrainingStrategies

local Others = script.Others

AqwamMachineDeepAndReinforcementLearningLibrary.Models = {

	LinearRegression = require(Models.LinearRegression),
	
	QuantileLinearRegression = require(Models.QuantileLinearRegression),
	
	PoissonLinearRegression = require(Models.PoissonLinearRegression),
	
	NormalLinearRegression = require(Models.NormalLinearRegression),
	
	BayesianLinearRegression = require(Models.BayesianLinearRegression),
	
	BayesianQuantileLinearRegression = require(Models.BayesianQuantileLinearRegression),
	
	PassiveAggressiveRegressor = require(Models.PassiveAggressiveRegressor),
	
	SupportVectorRegression = require(Models.SupportVectorRegression),
	
	KNearestNeighboursRegressor = require(Models.KNearestNeighboursRegressor),
	
	LogisticRegression = require(Models.LogisticRegression),
	
	PassiveAggressiveClassifier = require(Models.PassiveAggressiveClassifier),
	
	OneClassPassiveAggressiveClassifier = require(Models.OneClassPassiveAggressiveClassifier),
	
	NeuralNetwork = require(Models.NeuralNetwork),
	
	OneClassSupportVectorMachine = require(Models.OneClassSupportVectorMachine),
	
	SupportVectorMachine = require(Models.SupportVectorMachine),
	
	NearestCentroid = require(Models.NearestCentroid),
	
	KNearestNeighboursClassifier = require(Models.KNearestNeighboursClassifier),
	
	GaussianNaiveBayes = require(Models.GaussianNaiveBayes),
	
	MultinomialNaiveBayes = require(Models.MultinomialNaiveBayes),
	
	BernoulliNaiveBayes = require(Models.BernoulliNaiveBayes),
	
	ComplementNaiveBayes = require(Models.ComplementNaiveBayes),
	
	CategoricalNaiveBayes = require(Models.CategoricalNaiveBayes),
	
	KMeans = require(Models.KMeans),
	
	FuzzyCMeans = require(Models.FuzzyCMeans),
	
	KMedoids = require(Models.KMedoids),
	
	AgglomerativeHierarchical = require(Models.AgglomerativeHierarchical),
	
	ExpectationMaximization = require(Models.ExpectationMaximization),
	
	MeanShift = require(Models.MeanShift),
	
	AffinityPropagation = require(Models.AffinityPropagation),
	
	DensityBasedSpatialClusteringOfApplicationsWithNoise = require(Models.DensityBasedSpatialClusteringOfApplicationsWithNoise),
	
	BisectingCluster = require(Models.BisectingCluster),
	
	DeepQLearning = require(Models.DeepQLearning),

	DeepDoubleQLearningV1 = require(Models.DeepDoubleQLearningV1),

	DeepDoubleQLearningV2 = require(Models.DeepDoubleQLearningV2),
	
	DeepClippedDoubleQLearning = require(Models.DeepClippedDoubleQLearning),
	
	DeepStateActionRewardStateAction = require(Models.DeepStateActionRewardStateAction),
	
	DeepDoubleStateActionRewardStateActionV1 = require(Models.DeepDoubleStateActionRewardStateActionV1),
	
	DeepDoubleStateActionRewardStateActionV2 = require(Models.DeepDoubleStateActionRewardStateActionV2),
	
	DeepExpectedStateActionRewardStateAction = require(Models.DeepExpectedStateActionRewardStateAction),
	
	DeepDoubleExpectedStateActionRewardStateActionV1 = require(Models.DeepDoubleExpectedStateActionRewardStateActionV1),
	
	DeepDoubleExpectedStateActionRewardStateActionV2 = require(Models.DeepDoubleExpectedStateActionRewardStateActionV2),
	
	DeepMonteCarloControl = require(Models.DeepMonteCarloControl),

	DeepOffPolicyMonteCarloControl = require(Models.DeepOffPolicyMonteCarloControl),
	
	DeepTemporalDifference = require(Models.DeepTemporalDifference),
	
	REINFORCE = require(Models.REINFORCE),
	
	VanillaPolicyGradient = require(Models.VanillaPolicyGradient),
	
	ActorCritic = require(Models.ActorCritic),
	
	SoftActorCritic = require(Models.SoftActorCritic),
	
	AdvantageActorCritic = require(Models.AdvantageActorCritic),
	
	TemporalDifferenceActorCritic = require(Models.TemporalDifferenceActorCritic),
	
	ProximalPolicyOptimization = require(Models.ProximalPolicyOptimization),
	
	ProximalPolicyOptimizationClip = require(Models.ProximalPolicyOptimizationClip),
	
	DeepDeterministicPolicyGradient = require(Models.DeepDeterministicPolicyGradient),
	
	TwinDelayedDeepDeterministicPolicyGradient = require(Models.TwinDelayedDeepDeterministicPolicyGradient),
	
	TabularQLearning = require(Models.TabularQLearning),
	
	TabularClippedDoubleQLearning = require(Models.TabularClippedDoubleQLearning),
	
	TabularDoubleQLearningV1 = require(Models.TabularDoubleQLearningV1),
	
	TabularDoubleQLearningV2 = require(Models.TabularDoubleQLearningV2),

	TabularStateActionRewardStateAction = require(Models.TabularStateActionRewardStateAction),
	
	TabularDoubleStateActionRewardStateActionV1 = require(Models.TabularDoubleStateActionRewardStateActionV1),

	TabularDoubleStateActionRewardStateActionV2 = require(Models.TabularDoubleStateActionRewardStateActionV2),
	
	TabularExpectedStateActionRewardStateAction = require(Models.TabularExpectedStateActionRewardStateAction),
	
	TabularDoubleExpectedStateActionRewardStateActionV1 = require(Models.TabularDoubleExpectedStateActionRewardStateActionV1),

	TabularDoubleExpectedStateActionRewardStateActionV2 = require(Models.TabularDoubleExpectedStateActionRewardStateActionV2),
	
	TabularMonteCarloControl = require(Models.TabularMonteCarloControl),
	
	TabularOffPolicyMonteCarloControl = require(Models.TabularOffPolicyMonteCarloControl),
	
	TabularTemporalDifference = require(Models.TabularTemporalDifference),
	
	Markov = require(Models.Markov),
	
	DynamicBayesianNetwork = require(Models.DynamicBayesianNetwork),
	
	ConditionalRandomField = require(Models.ConditionalRandomField),
	
	KalmanFilter = require(Models.KalmanFilter),
	
	ExtendedKalmanFilter = require(Models.ExtendedKalmanFilter),
	
	UnscentedKalmanFilter = require(Models.UnscentedKalmanFilter),
	
	UnscentedKalmanFilterDataPredictVariant = require(Models.UnscentedKalmanFilterDataPredictVariant),
	
	GenerativeAdversarialNetwork = require(Models.GenerativeAdversarialNetwork),
	
	ConditionalGenerativeAdversarialNetwork = require(Models.ConditionalGenerativeAdversarialNetwork),

	WassersteinGenerativeAdversarialNetwork = require(Models.WassersteinGenerativeAdversarialNetwork),

	ConditionalWassersteinGenerativeAdversarialNetwork = require(Models.ConditionalWassersteinGenerativeAdversarialNetwork),

}

AqwamMachineDeepAndReinforcementLearningLibrary.Regularizers = {
	
	ElasticNet = require(Regularizers.ElasticNet),
	
	Lasso = require(Regularizers.Lasso),
	
	Ridge = require(Regularizers.Ridge),
	
}

AqwamMachineDeepAndReinforcementLearningLibrary.Optimizers = {

	AdaptiveDelta = require(Optimizers.AdaptiveDelta),
	
	AdaptiveFactor = require(Optimizers.AdaptiveFactor),
	
	AdaptiveGradient = require(Optimizers.AdaptiveGradient),

	AdaptiveMomentEstimation = require(Optimizers.AdaptiveMomentEstimation),

	AdaptiveMomentEstimationMaximum = require(Optimizers.AdaptiveMomentEstimationMaximum),
	
	AdaptiveMomentEstimationWeightDecay = require(Optimizers.AdaptiveMomentEstimationWeightDecay),
	
	Gravity = require(Optimizers.Gravity),
	
	Momentum = require(Optimizers.Momentum),

	NesterovAcceleratedAdaptiveMomentEstimation = require(Optimizers.NesterovAcceleratedAdaptiveMomentEstimation),
	
	RectifiedAdaptiveMomentEstimation = require(Optimizers.RectifiedAdaptiveMomentEstimation),
	
	ResilientBackwardPropagation = require(Optimizers.ResilientBackwardPropagation),
	
	RootMeanSquarePropagation = require(Optimizers.RootMeanSquarePropagation),

}

AqwamMachineDeepAndReinforcementLearningLibrary.ValueSchedulers = {
	
	Chained = require(ValueSchedulers.Chained),
	
	Constant = require(ValueSchedulers.Constant),
	
	CosineAnnealing = require(ValueSchedulers.CosineAnnealing),
	
	Exponential = require(ValueSchedulers.Exponential),
	
	InverseSquareRoot = require(ValueSchedulers.InverseSquareRoot),
	
	InverseTime = require(ValueSchedulers.InverseTime),
	
	Linear = require(ValueSchedulers.Linear),
	
	MultipleStep = require(ValueSchedulers.MultipleStep),
	
	Multiplicative = require(ValueSchedulers.Multiplicative),
	
	Polynomial = require(ValueSchedulers.Polynomial),
	
	Sequential = require(ValueSchedulers.Sequential),

	Step = require(ValueSchedulers.Step),

}

AqwamMachineDeepAndReinforcementLearningLibrary.GradientClippers = {

	ClipValue = require(GradientClippers.ClipValue),

	ClipNormalization = require(GradientClippers.ClipNormalization),
	
}

AqwamMachineDeepAndReinforcementLearningLibrary.ExperienceReplays = {

	UniformExperienceReplay = require(ExperienceReplays.UniformExperienceReplay),

	PrioritizedExperienceReplay = require(ExperienceReplays.PrioritizedExperienceReplay),

	NStepExperienceReplay = require(ExperienceReplays.NStepExperienceReplay),

}

AqwamMachineDeepAndReinforcementLearningLibrary.QuickSetups = {

	SingleCategoricalPolicy = require(QuickSetups.SingleCategoricalPolicy),

	SingleDiagonalGaussianPolicy = require(QuickSetups.SingleDiagonalGaussianPolicy),
	
	QueuedCategoricalPolicy = require(QuickSetups.QueuedCategoricalPolicy),

	QueuedDiagonalGaussianPolicy = require(QuickSetups.QueuedDiagonalGaussianPolicy),
	
	ParallelCategoricalPolicy = require(QuickSetups.ParallelCategoricalPolicy),

	ParallelDiagonalGaussianPolicy = require(QuickSetups.ParallelDiagonalGaussianPolicy),

}

AqwamMachineDeepAndReinforcementLearningLibrary.EligibilityTraces = {
	
	AccumulatingTrace = require(EligibilityTraces.AccumulatingTrace),
	
	ReplacingTrace = require(EligibilityTraces.ReplacingTrace),
	
	DutchTrace = require(EligibilityTraces.DutchTrace),
	
}

AqwamMachineDeepAndReinforcementLearningLibrary.ReinforcementLearningStrategies = {
	
	RandomNetworkDistillation = require(ReinforcementLearningStrategies.RandomNetworkDistillation),
	
	GenerativeAdversarialImitationLearning = require(ReinforcementLearningStrategies.GenerativeAdversarialImitationLearning),

	WassersteinGenerativeAdversarialImitationLearning = require(ReinforcementLearningStrategies.WassersteinGenerativeAdversarialImitationLearning),
	
}

AqwamMachineDeepAndReinforcementLearningLibrary.DistributedTrainingStrategies = {
	
	DistributedGradientsCoordinator = require(DistributedTrainingStrategies.DistributedGradientsCoordinator),

	DistributedModelParametersCoordinator = require(DistributedTrainingStrategies.DistributedModelParametersCoordinator),
	
}

AqwamMachineDeepAndReinforcementLearningLibrary.Others = {
	
	NormalModelModifier = require(Others.NormalModelModifier),
	
	ModelTrainingModifier = require(Others.ModelTrainingModifier),
	
	ModelSafeguardWrapper = require(Others.ModelSafeguardWrapper),
	
	ModelParametersMerger = require(Others.ModelParametersMerger),

	ModelChecker = require(Others.ModelChecker),
	
	OneVsAll = require(Others.OneVsAll),
	
	OneVsOne = require(Others.OneVsOne),

	OnlineLearning = require(Others.OnlineLearning),
	
	DatasetCreator = require(Others.DatasetCreator),
	
	ConfusionMatrixCreator = require(Others.ConfusionMatrixCreator),

}

return AqwamMachineDeepAndReinforcementLearningLibrary
