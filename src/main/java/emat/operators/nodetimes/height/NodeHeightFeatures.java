package emat.operators.nodetimes.height;

/**
 * The input features of the predictor of the conditional density of a node height. The node lies between its parent and
 * its older child. The branch a is the one from the parent to the node, b from the node to its older child and c from
 * the node to its younger child.
 */
public final class NodeHeightFeatures {

    /** The number of features that {@link #compute} returns. */
    public static final int COUNT = 26;

    private NodeHeightFeatures() {
    }

    /** Returns the features of the given height. */
    public static double[] compute(NodeHeightContext context, double height) {
        double lengthA = context.parentHeight() - height;
        double lengthB = height - context.olderChildHeight();
        double lengthC = height - context.youngerChildHeight();
        double logLengthA = Math.log(lengthA);
        double logLengthB = Math.log(lengthB);
        double logLengthC = Math.log(lengthC);
        double logPopSize = Math.log(context.popSize());

        return new double[]{
                context.parentHeight(), context.olderChildHeight(), context.youngerChildHeight(), height,
                // the relative position between the older child and the parent
                (height - context.olderChildHeight()) / (context.parentHeight() - context.olderChildHeight()),
                lengthA, lengthB, lengthC,
                context.numParentMutations(), context.numOlderMutations(), context.numYoungerMutations(),
                logPopSize, context.growthRate(), context.clockRate(), context.kappa(),
                logLengthA, logLengthB, logLengthC,
                context.numParentMutations() * logLengthA, context.numOlderMutations() * logLengthB, context.numYoungerMutations() * logLengthC,
                context.clockRate() * lengthA, context.clockRate() * lengthB, context.clockRate() * lengthC,
                // the log population size of the exponential growth model at the height
                logPopSize - context.growthRate() * height,
                // the number of lineages at the height of the parent, which sets the rate of coalescence there
                context.numParentLineages()
        };
    }

}
