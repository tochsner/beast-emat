import emat.adapters.MutationCountAdapter;
import emat.alignment.FitchFilteredAlignment;
import emat.benchmark.TimedMCMC;
import emat.initalisation.ParsimonyMutationsInitialiser;
import emat.operators.BranchReformOperator;
import emat.operators.IdenticalTipPermutationOperator;
import emat.operators.IntervalScaleOperator;
import emat.operators.MutationDirectedSprOperator;
import emat.operators.MutationTimeOperator;
import emat.operators.InnerNodeDisplacementOperator;
import emat.operators.BatchNodeDisplacementOperator;
import emat.operators.SiteHistoryGibbsOperator;
import emat.operators.SiteRegionGibbsOperator;
import emat.operators.RootScaleOperator;
import emat.operators.ParameterFollowingOperator;
import emat.helper.ExcessMutationLogger;
import emat.helper.ParameterFollowing;
import emat.helper.TreePriorSurrogate;
import emat.prior.GeneticPrior;
import emat.state.Mutations;
import emat.tree.IncrementalStoreTree;
import emat.tree.InsertionSortTreeIntervals;

open module my.beast.example {
    requires beast.pkgmgmt;
    requires beast.base;
    requires org.apache.commons.statistics.distribution;
    requires org.apache.commons.math4.legacy;
    requires java.xml;
    requires adaptiveoperators;

    exports emat.adapters;
    exports emat.alignment;
    exports emat.benchmark;
    exports emat.prior;
    exports emat.operators;
    exports emat.helper;
    exports emat.initalisation;
    exports emat.state;
    exports emat.stochasticmapping;
    exports emat.tree;

    provides beast.base.core.BEASTInterface with
            Mutations,
            FitchFilteredAlignment,
            GeneticPrior,
            ParsimonyMutationsInitialiser,
            MutationTimeOperator,
            InnerNodeDisplacementOperator,
            BatchNodeDisplacementOperator,
            SiteHistoryGibbsOperator,
            SiteRegionGibbsOperator,
            RootScaleOperator,
            IntervalScaleOperator,
            MutationDirectedSprOperator,
            IdenticalTipPermutationOperator,
            BranchReformOperator,
            ParameterFollowingOperator,
            ParameterFollowing,
            ExcessMutationLogger,
            TreePriorSurrogate,
            InsertionSortTreeIntervals,
            IncrementalStoreTree,
            MutationCountAdapter,
            TimedMCMC;
}
