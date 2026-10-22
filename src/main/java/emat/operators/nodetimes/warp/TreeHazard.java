package emat.operators.nodetimes.warp;

/**
 * The rate at which a tree prior produces its branching events, as the number of units at risk times a rate per unit
 * that only depends on the height and the parameters of the prior.
 */
public enum TreeHazard {

    /** Every pair of lineages merges at the same rate. */
    COALESCENT,

    /** Every lineage branches at the same rate. */
    BIRTH_DEATH,

    /** Is chosen at the first fit as the one of the two above whose model leaves the smaller residual. */
    AUTO;

    /** Returns the number of units at risk while the given number of lineages is present. */
    public double computeExposureMultiplicity(int numLineages) {
        return switch (this) {
            case COALESCENT -> numLineages * (numLineages - 1) / 2.0;
            case BIRTH_DEATH -> numLineages;
            case AUTO -> throw new IllegalStateException("The hazard AUTO has to be resolved first.");
        };
    }

    /** Returns the number of units at risk for the branching event that leaves the given number of lineages below it. */
    public double computeEventMultiplicity(int numLineagesBelow) {
        return switch (this) {
            case COALESCENT -> numLineagesBelow * (numLineagesBelow - 1) / 2.0;
            case BIRTH_DEATH -> numLineagesBelow - 1;
            case AUTO -> throw new IllegalStateException("The hazard AUTO has to be resolved first.");
        };
    }

}
