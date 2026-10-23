package emat.state;

/**
 * A single mutation at one site on the branch above the given node. Both times are BEAST
 * node heights, which grow backwards in time. Evolution runs forwards in time from the
 * parent to the node, so oldState is the state closer to the root and time is smaller than
 * timeOfPreviousMutation. The first mutation at a site on a branch has no previous mutation
 * and stores {@link #BRANCH_START} as timeOfPreviousMutation, so that moving the start of the
 * branch does not require changing any mutation.
 */
public record Mutation(int nodeNr, double time, double timeOfPreviousMutation, int site, int oldState, int newState)  {

    /** The time of the previous mutation of the first mutation at a site on a branch. */
    public static final double BRANCH_START = Double.NaN;

    /** Returns whether this is the first mutation at its site on its branch. */
    public boolean isFirstAtSite() {
        return Double.isNaN(this.timeOfPreviousMutation);
    }

    /** Returns the height at which the site was last mutated, or the given branch start if this is the first mutation at its site. */
    public double getTimeOfPreviousMutation(double branchStartHeight) {
        return this.isFirstAtSite() ? branchStartHeight : this.timeOfPreviousMutation;
    }

}
