package emat.helper;

/**
 * A local approximation of the log tree prior as a function of the height of a single node, learned from observed
 * changes of the log tree prior. An observation is made of one or several nodes that moved at once within the
 * ranges they can take, and of the change of the log tree prior this caused. The approximation explains the
 * change as the sum of the contributions of the moved nodes.
 */
public interface LearnedTreePrior {

    /**
     * Returns the slope of the log tree prior in the height of a node at the given midpoint of its range, for the
     * given values of the parameters of the tree prior. The slope is zero until the first fit.
     */
    double getSlope(double midpoint, double[] parameters);

    /** Starts a new observation, which drops the nodes of an observation that was not finished. */
    void startObservation();

    /**
     * Adds a node to the current observation that moved from the old to the new offset from the midpoint of its
     * range.
     */
    void addNode(double midpoint, double[] parameters, double oldOffset, double newOffset);

    /**
     * Finishes the current observation with the observed change of the log tree prior. An observation without
     * nodes or with a target that is not finite is dropped.
     */
    void finishObservation(double target);

    /** Drops the current observation. */
    void discardObservation();

    boolean isFitted();

    long getNumObservations();

}
