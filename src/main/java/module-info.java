import emat.alignment.FitchFilteredAlignment;
import emat.initalisation.ParsimonyMutationsInitialiser;
import emat.operators.BranchReformOperator;
import emat.operators.IntervalScaleOperator;
import emat.operators.MutationDirectedSprOperator;
import emat.operators.MutationTimeOperator;
import emat.operators.InnerNodeDisplacementOperator;
import emat.operators.SiteHistoryGibbsOperator;
import emat.operators.SubtreeSlideOperator;
import emat.operators.WilsonBaldingOperator;
import emat.operators.RootScaleOperator;
import emat.prior.GeneticPrior;
import emat.state.Mutations;

open module my.beast.example {
    requires beast.pkgmgmt;
    requires beast.base;
    requires org.apache.commons.statistics.distribution;
    requires java.xml;

    exports emat.alignment;
    exports emat.prior;
    exports emat.operators;
    exports emat.helper;
    exports emat.initalisation;
    exports emat.state;

    provides beast.base.core.BEASTInterface with
            Mutations,
            FitchFilteredAlignment,
            GeneticPrior,
            ParsimonyMutationsInitialiser,
            MutationTimeOperator,
            InnerNodeDisplacementOperator,
            SiteHistoryGibbsOperator,
            RootScaleOperator,
            SubtreeSlideOperator,
            WilsonBaldingOperator,
            IntervalScaleOperator,
            MutationDirectedSprOperator,
            BranchReformOperator;
}
