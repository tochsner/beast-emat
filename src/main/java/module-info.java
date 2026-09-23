import emat.GeneticPrior;

open module my.beast.example {
    requires beast.pkgmgmt;
    requires beast.base;
    requires org.apache.commons.statistics.distribution;
    requires java.xml;

    exports emat;

    provides beast.base.core.BEASTInterface with
        emat.Mutations,
            GeneticPrior,
            emat.ParsimonyMutationsInitialiser,
            emat.MutationTimeOperator,
            emat.SiteHistoryGibbsOperator;
}
