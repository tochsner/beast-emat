import emat.adapters.MutationCountAdapter;
import emat.adapters.RatioPowerAdapter;
import emat.adapters.RootRatioAdapter;
import emat.adapters.TreeIntervalAdapter;
import emat.alignment.FitchFilteredAlignment;
import emat.alignment.MissingSiteFilteredAlignment;
import emat.benchmark.ExactHistoryInitialiser;
import emat.benchmark.HistoryDistanceLogger;
import emat.benchmark.TimedMCMC;
import emat.initalisation.ParsimonyMutationsInitialiser;
import emat.operators.mutations.BranchReformOperator;
import emat.operators.nodetimes.warp.LearnedWarpOperator;
import emat.operators.nodetimes.IidIntervalScaleOperator;
import emat.operators.nodetimes.BandedIntervalScaleOperator;
import emat.operators.nodetimes.MutationBandedIntervalScaleOperator;
import emat.operators.nodetimes.MutationNoiseIntervalScaleOperator;
import emat.operators.nodetimes.IntervalScaleOperator;
import emat.operators.treeprior.LearnedTreePriorOperator;
import emat.operators.treeprior.TreePriorResamplingOperator;
import emat.operators.nodetimes.LiftedIntervalScaleOperator;
import emat.operators.nodetimes.TiltProbeOperator;
import emat.operators.nodetimes.TiltedIntervalScaleOperator;
import emat.operators.topology.MutationDirectedSprOperator;
import emat.operators.mutations.MutationTimeOperator;
import emat.operators.nodetimes.InnerNodeDisplacementOperator;
import emat.operators.nodetimes.batch.BatchNodeDisplacementOperator;
import emat.operators.nodetimes.batch.MappedBatchNodeDisplacementOperator;
import emat.operators.nodetimes.height.LearnedBatchNodeDisplacementOperator;
import emat.operators.nodetimes.height.NodeHeightLearner;
import emat.operators.nodetimes.height.NodeHeightTrainer;
import emat.operators.nodetimes.batch.GaussianRotationNodeDisplacementOperator;
import emat.operators.nodetimes.batch.QuantileIntervalScaleOperator;
import emat.operators.nodetimes.batch.QuantileShiftNodeDisplacementOperator;
import emat.operators.mutations.SiteHistoryGibbsOperator;
import emat.operators.mutations.MissingCladeGibbsOperator;
import emat.operators.nodetimes.RootScaleOperator;
import emat.operators.nodetimes.SequentialMoveOperator;
import emat.operators.nodetimes.RatioScaleOperator;
import emat.operators.nodetimes.GammaRegressionRatioScaleOperator;
import emat.operators.nodetimes.GammaTransportRatioScaleOperator;
import emat.operators.nodetimes.TipOffsetScaleOperator;
import emat.operators.nodetimes.GapShiftScaleOperator;
import emat.operators.nodetimes.PresentRatioScaleOperator;
import emat.operators.nodetimes.YoungestTipRatioScaleOperator;
import emat.operators.nodetimes.RootAnchoredRatioScaleOperator;
import emat.operators.nodetimes.MutationRatioScaleOperator;
import emat.operators.nodetimes.RootRatioScaleOperator;
import emat.operators.nodetimes.PowerRatioScaleOperator;
import emat.operators.nodetimes.GammaRootRatioScaleOperator;
import emat.diagnostics.ChainDiagnostics;
import emat.diagnostics.DiagnosticMCMC;
import emat.helper.ExcessMutationLogger;
import emat.prior.GeneticPrior;
import emat.state.Mutations;
import emat.tree.IncrementalStoreTree;
import emat.tree.InsertionSortTreeIntervals;

open module my.beast.example {
    requires beast.pkgmgmt;
    requires beast.base;
    requires org.apache.commons.statistics.distribution;
    requires org.apache.commons.math4.legacy;
    requires ojalgo;
    requires java.xml;
    requires adaptiveoperators;

    exports emat.adapters;
    exports emat.alignment;
    exports emat.benchmark;
    exports emat.diagnostics;
    exports emat.prior;
    exports emat.operators.mutations;
    exports emat.operators.nodetimes;
    exports emat.operators.nodetimes.batch;
    exports emat.operators.nodetimes.height;
    exports emat.operators.nodetimes.warp;
    exports emat.operators.topology;
    exports emat.operators.treeprior;
    exports emat.operators.topology.mdspr;
    exports emat.helper;
    exports emat.initalisation;
    exports emat.state;
    exports emat.stochasticmapping;
    exports emat.tree;

    provides beast.base.core.BEASTInterface with
            Mutations,
            FitchFilteredAlignment,
            MissingSiteFilteredAlignment,
            GeneticPrior,
            ParsimonyMutationsInitialiser,
            MutationTimeOperator,
            InnerNodeDisplacementOperator,
            BatchNodeDisplacementOperator,
            MappedBatchNodeDisplacementOperator,
            LearnedBatchNodeDisplacementOperator,
            NodeHeightLearner,
            NodeHeightTrainer,
            GaussianRotationNodeDisplacementOperator,
            QuantileIntervalScaleOperator,
            QuantileShiftNodeDisplacementOperator,
            SiteHistoryGibbsOperator,
            MissingCladeGibbsOperator,
            RootScaleOperator,
            SequentialMoveOperator,
            RatioScaleOperator,
            GammaRegressionRatioScaleOperator,
            GammaTransportRatioScaleOperator,
            TipOffsetScaleOperator,
            GapShiftScaleOperator,
            PresentRatioScaleOperator,
            YoungestTipRatioScaleOperator,
            RootAnchoredRatioScaleOperator,
            MutationRatioScaleOperator,
            RootRatioScaleOperator,
            PowerRatioScaleOperator,
            GammaRootRatioScaleOperator,
            IidIntervalScaleOperator,
            BandedIntervalScaleOperator,
            MutationBandedIntervalScaleOperator,
            MutationNoiseIntervalScaleOperator,
            IntervalScaleOperator,
            LearnedTreePriorOperator,
            TreePriorResamplingOperator,
            LiftedIntervalScaleOperator,
            TiltProbeOperator,
            TiltedIntervalScaleOperator,
            LearnedWarpOperator,
            MutationDirectedSprOperator,
            BranchReformOperator,
            ExcessMutationLogger,
            InsertionSortTreeIntervals,
            IncrementalStoreTree,
            MutationCountAdapter,
            TreeIntervalAdapter,
            RootRatioAdapter,
            RatioPowerAdapter,
            TimedMCMC,
            ChainDiagnostics,
            DiagnosticMCMC,
            HistoryDistanceLogger,
            ExactHistoryInitialiser;
}
