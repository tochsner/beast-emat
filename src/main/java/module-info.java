open module my.beast.example {
    requires beast.pkgmgmt;
    requires beast.base;
    requires org.apache.commons.statistics.distribution;
    requires java.xml;

    exports emat;

    provides beast.base.core.BEASTInterface with
        my.beast.example.MyDistribution,
        my.beast.example.MyScaleOperator;
}
