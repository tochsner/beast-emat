import emat.initalisation.ParsimonyMutationsInitialiser;
import emat.operators.IntervalScaleOperator;
import emat.operators.MutationDirectedSprOperator;
import emat.operators.MutationTimeOperator;
import emat.operators.SiteHistoryGibbsOperator;
import emat.operators.SubtreeSlideOperator;
import emat.operators.WilsonBaldingOperator;
import emat.prior.GeneticPrior;
import emat.state.Mutations;

open module my.beast.example {
    requires beast.pkgmgmt;
    requires beast.base;
    requires org.apache.commons.statistics.distribution;
    requires java.xml;

    exports emat.prior;
    exports emat.operators;
    exports emat.helper;
    exports emat.initalisation;
    exports emat.state;

    provides beast.base.core.BEASTInterface with
            Mutations,
            GeneticPrior,
            ParsimonyMutationsInitialiser,
            MutationTimeOperator,
            SiteHistoryGibbsOperator,
            SubtreeSlideOperator,
            WilsonBaldingOperator,
            IntervalScaleOperator,
            MutationDirectedSprOperator;
}
