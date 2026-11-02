package emat.operators.nodetimes.height;

/**
 * What the conditional density of the height of an internal node depends on, apart from the height itself: the heights of
 * its parent and its two children, the numbers of mutations on the three branches around it, and the parameters of the
 * tree prior and the substitution model.
 */
public record NodeHeightContext(double parentHeight, double olderChildHeight, double youngerChildHeight,
                                int numParentMutations, int numOlderMutations, int numYoungerMutations,
                                double popSize, double growthRate, double clockRate, double kappa, int numParentLineages) {
}
