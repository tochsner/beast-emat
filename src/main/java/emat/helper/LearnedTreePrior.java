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

    /**
     * Returns the quadratic coefficient c_2 of the log tree prior in the offset of a node from the midpoint of its
     * range, for the given values of the parameters of the tree prior. The curvature is zero until the first fit.
     */
    double getCurvature(double midpoint, double[] parameters);

    /**
     * Draws a height between the lower and the upper bound from the density ∝ exp(-decayRate h + a(h)), where a(h)
     * is the approximation of the log tree prior as a function of the height of the node. By default, a is the
     * quadratic polynomial in the offset from the midpoint of the bounds that is given by the learned slope and
     * curvature there, and only a downward curvature is kept so that the density stays log-concave.
     */
    default double sampleHeight(double decayRate, double lower, double upper, double[] parameters) {
        double midpoint = (lower + upper) / 2.0;
        double slope = this.getSlope(midpoint, parameters);
        double stiffness = -this.getUsableCurvature(lower, upper, parameters);

        if (stiffness == 0.0) {
            return TruncatedDensitySampler.sampleExponential(decayRate - slope, lower, upper);
        }

        double offset = TruncatedDensitySampler.sampleGaussianTilt(decayRate - slope, stiffness, lower - midpoint, upper - midpoint);
        return Math.min(Math.max(midpoint + offset, lower), upper);
    }

    /**
     * Returns a(newHeight) - a(oldHeight) for the approximation a of the log tree prior that
     * {@link #sampleHeight} draws from, for a node that moves between the lower and the upper bound.
     */
    default double getChange(double lower, double upper, double oldHeight, double newHeight, double[] parameters) {
        double midpoint = (lower + upper) / 2.0;
        double slope = this.getSlope(midpoint, parameters);
        double curvature = this.getUsableCurvature(lower, upper, parameters);

        double oldOffset = oldHeight - midpoint;
        double newOffset = newHeight - midpoint;
        return slope * (newOffset - oldOffset) + curvature * (newOffset * newOffset - oldOffset * oldOffset);
    }

    /**
     * Returns the curvature that the default proposal uses between the bounds: zero if it is not negative, and
     * zero if it is negligible over the range, so that sampling and density agree.
     */
    private double getUsableCurvature(double lower, double upper, double[] parameters) {
        double curvature = Math.min(this.getCurvature((lower + upper) / 2.0, parameters), 0.0);
        double halfWidth = (upper - lower) / 2.0;
        return -curvature * halfWidth * halfWidth < 1e-9 ? 0.0 : curvature;
    }

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
