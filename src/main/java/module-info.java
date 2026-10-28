import emat.adapters.MutationCountAdapter;
import emat.alignment.FitchFilteredAlignment;
import emat.alignment.MissingSiteFilteredAlignment;
import emat.benchmark.ExactHistoryInitialiser;
import emat.benchmark.HistoryDistanceLogger;
import emat.benchmark.TimedMCMC;
import emat.initalisation.ParsimonyMutationsInitialiser;
import emat.operators.mutations.BranchReformOperator;
import emat.operators.mutations.StarReformOperator;
import emat.operators.nodetimes.warp.LearnedWarpOperator;
import emat.operators.nodetimes.IntervalScaleOperator;
import emat.operators.topology.MutationDirectedSprOperator;
import emat.operators.mutations.MutationTimeOperator;
import emat.operators.nodetimes.InnerNodeDisplacementOperator;
import emat.operators.nodetimes.batch.BatchNodeDisplacementOperator;
import emat.operators.mutations.SiteHistoryGibbsOperator;
import emat.operators.mutations.MissingCladeGibbsOperator;
import emat.operators.nodetimes.RootScaleOperator;
import emat.operators.nodetimes.RatioScaleOperator;
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
    exports emat.prior;
    exports emat.operators.mutations;
    exports emat.operators.nodetimes;
    exports emat.operators.nodetimes.batch;
    exports emat.operators.nodetimes.warp;
    exports emat.operators.topology;
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
            SiteHistoryGibbsOperator,
            MissingCladeGibbsOperator,
            RootScaleOperator,
            RatioScaleOperator,
            IntervalScaleOperator,
            LearnedWarpOperator,
            MutationDirectedSprOperator,
            BranchReformOperator,
            StarReformOperator,
            ExcessMutationLogger,
            InsertionSortTreeIntervals,
            IncrementalStoreTree,
            MutationCountAdapter,
            TimedMCMC,
            HistoryDistanceLogger,
            ExactHistoryInitialiser;
}
