package emat.operators.nodetimes.batch;

import beast.base.core.Description;
import beast.base.evolution.tree.Node;
import beast.base.util.Randomizer;
import emat.operators.nodetimes.BranchTimeMapper;
import org.apache.commons.statistics.distribution.GammaDistribution;
import org.apache.commons.statistics.distribution.NormalDistribution;

@Description("Moves the intervals between every internal node and its older child by a shared shift of their quantiles, " +
        "and maps every mutation linearly within its branch. The interval scale operator multiplies all intervals by the " +
        "same factor, although they differ in how far the data allow them to move. Here, every interval x has a reference " +
        "density ∝ (x + s)^m exp(-b x) on x > 0, which approximates its conditional density. The power m is the number of " +
        "mutations on the two child branches, whose lengths grow with x, and s is their mean gap to the older child. The " +
        "rate b is the rate at which these branches leave their states, reduced by the slope of a learned approximation " +
        "of the log tree prior. The quantile of each interval is transported to a normal score, which is shifted by the same " +
        "amount on a circle between the largest scores, and mapped back. Intervals that the data pin down then move little, and loose ones far. An interval stays as it is " +
        "if its normal score is among the extreme ones of its reference, where a wrong rate would make the move implausible. The reference of an interval depends on the heights of its descendants through the " +
        "means of their old and new values, which are the same when the move is reversed, so that the proposal is " +
        "symmetric. It changes the tree length, but not the clock rate.")
public class QuantileIntervalScaleOperator extends TransportedNodeDisplacementOperator {

    // intervals with a normal score of larger size stay as they are, which is a tail of 1e-9
    private static final double DEFAULT_MAX_SCORE = 6.0;

    private static final NormalDistribution NORMAL = NormalDistribution.of(0.0, 1.0);

    // the shift of the normal scores is uniform up to this size, at which it covers the whole circle between the largest scores
    private static final double MAX_STEP_SIZE = DEFAULT_MAX_SCORE;

    // raising an interval raises the node and its ancestors that follow it as older child, which are two nodes on average
    private static final double MEAN_NUM_RAISED_NODES = 2.0;

    // the share of the rate of the genetic prior that the tree prior cannot remove
    private static final double MIN_RATE_SHARE = 0.1;

    // the step size is not tuned on proposals in which fewer than this share of the intervals moved
    private static final double MIN_MOVED_SHARE = 0.5;

    // the heights before the proposal, by node number
    private double[] oldHeights;

    // the shift of the normal scores and the part of the log Hastings ratio of the intervals so far
    private double scoreShift;
    private double logHastingsRatio;

    // the number of intervals of the last proposal, and the number of those that moved
    private int numIntervals;
    private int numMovedIntervals;

    // the reference density of the interval of the current node: the power m, the offset s and the rate b of (x + s)^m exp(-b x)
    private double power;
    private double offset;
    private double rate;

    // the height at which the slope of the tree prior is taken for the current node
    private double treePriorHeight;
    private boolean hasReference;

    @Override
    public void initAndValidate() {
        super.initAndValidate();
        this.oldHeights = new double[this.tree.getNodeCount()];
    }

    /**
     * Shifts the normal scores of all intervals by the same amount and maps the mutations along with the branches. The
     * log Hastings ratio is the sum over the intervals of the log ratio of the reference density at the old and at
     * the new interval, which is the log Jacobian of the shift, and the log Jacobian of the mapping of the mutations.
     */
    @Override
    protected double moveNodes() {
        Node[] nodes = this.tree.getNodesAsArray();
        for (Node node : nodes) {
            this.oldHeights[node.getNr()] = node.getHeight();
        }

        this.scoreShift = this.drawScoreShift();
        this.logHastingsRatio = 0.0;
        this.numIntervals = 0;
        this.numMovedIntervals = 0;
        this.shiftIntervals(this.tree.getRoot());

        return this.logHastingsRatio + BranchTimeMapper.mapMutations(this.mutations, nodes, this.oldHeights, this);
    }

    /** Draws the shift of the normal scores, which is symmetric around zero. */
    protected double drawScoreShift() {
        return this.getCoercableParameterValue() * (2.0 * Randomizer.nextDouble() - 1.0);
    }

    /** Shifts the normal score of the interval between the given node and its older child after doing so for everything below it. */
    private void shiftIntervals(Node node) {
        if (node.isLeaf()) {
            return;
        }

        Node left = node.getLeft();
        Node right = node.getRight();
        this.shiftIntervals(left);
        this.shiftIntervals(right);

        // the descendants have their new heights now; the old ones are remembered

        double oldLeftHeight = this.oldHeights[left.getNr()];
        double oldRightHeight = this.oldHeights[right.getNr()];
        double newLeftHeight = left.getHeight();
        double newRightHeight = right.getHeight();

        double oldMaxHeight = Math.max(oldLeftHeight, oldRightHeight);
        double newMaxHeight = Math.max(newLeftHeight, newRightHeight);
        double oldInterval = this.oldHeights[node.getNr()] - oldMaxHeight;

        this.numIntervals++;
        this.hasReference = false;
        double newInterval = this.computeNewInterval(node, oldInterval, oldLeftHeight, oldRightHeight, newLeftHeight, newRightHeight);
        double newHeight = newMaxHeight + newInterval;
        node.setHeight(newHeight);

        // remember the move for the fit of the approximation of the tree prior, also if the interval stayed

        LearnedTreePrior treePrior = this.getTreePriorApproximation();
        if (treePrior != null && this.hasReference) {
            treePrior.addNode(this.treePriorHeight, this.getTreePriorParameters(),
                    this.oldHeights[node.getNr()] - this.treePriorHeight, newHeight - this.treePriorHeight);
        }
    }

    /**
     * Returns the interval after the shift of its normal score, or the old interval if it stays. The reference depends on
     * the means of the old and the new values of the heights below the node, which are the same in the reverse move.
     */
    private double computeNewInterval(Node node, double oldInterval, double oldLeftHeight, double oldRightHeight,
                                      double newLeftHeight, double newRightHeight) {
        double oldMaxHeight = Math.max(oldLeftHeight, oldRightHeight);
        double newMaxHeight = Math.max(newLeftHeight, newRightHeight);

        double anchorHeight = (oldMaxHeight + newMaxHeight) / 2.0;
        double leftGap = ((oldMaxHeight - oldLeftHeight) + (newMaxHeight - newLeftHeight)) / 2.0;
        double rightGap = ((oldMaxHeight - oldRightHeight) + (newMaxHeight - newRightHeight)) / 2.0;

        if (!this.computeReference(node, anchorHeight, leftGap, rightGap) || !(oldInterval > 0.0)) {
            return oldInterval;
        }

        this.hasReference = true;

        // the reference is a gamma density of x + s from s on, whose tails are read from the side where they are small

        GammaDistribution gamma = GammaDistribution.of(this.power + 1.0, 1.0 / this.rate);
        double offsetSurvival = gamma.survivalProbability(this.offset);
        if (!(offsetSurvival > 0.0)) {
            return oldInterval;
        }
        double lowerTail = Math.min(gamma.probability(this.offset, this.offset + oldInterval) / offsetSurvival, 1.0);
        double upperTail = Math.min(gamma.survivalProbability(this.offset + oldInterval) / offsetSurvival, 1.0);
        if (Double.isNaN(lowerTail) || Double.isNaN(upperTail)) {
            return oldInterval;
        }
        double oldScore = lowerTail < 0.5 ? NORMAL.inverseCumulativeProbability(lowerTail) : -NORMAL.inverseCumulativeProbability(upperTail);

        double maxScore = this.getMaxScore();
        if (!(Math.abs(oldScore) < maxScore)) {
            return oldInterval;
        }
        double newScore = this.shiftScore(oldScore, maxScore);

        double newUpperTail = NORMAL.cumulativeProbability(-newScore);
        double newInterval = gamma.inverseSurvivalProbability(newUpperTail * offsetSurvival) - this.offset;
        if (!(newInterval > 0.0) || Double.isInfinite(newInterval)) {
            return oldInterval;
        }

        this.numMovedIntervals++;

        // the density of the reference at the old interval over that at the new one, and the density of the normal score at the new score over that at the old one, is the Jacobian of the move

        this.logHastingsRatio += this.power * (Math.log(oldInterval + this.offset) - Math.log(newInterval + this.offset))
                - this.rate * (oldInterval - newInterval)
                + (oldScore * oldScore - newScore * newScore) / 2.0;

        LearnedTreePrior treePrior = this.getTreePriorApproximation();
        if (treePrior != null) {
            double[] parameters = this.getTreePriorParameters();
            double slope = treePrior.getSlope(this.treePriorHeight, parameters);
            this.addApproximateTreePriorChange(MEAN_NUM_RAISED_NODES * slope * (newInterval - oldInterval));
        }

        return newInterval;
    }

    /** Returns the largest absolute normal score of an interval that can move. */
    protected double getMaxScore() {
        return DEFAULT_MAX_SCORE;
    }

    /**
     * Shifts the normal score on the circle between the largest scores. It stays between them, so that the reverse
     * move, which starts there, leaves the same intervals as they are.
     */
    private double shiftScore(double score, double maxScore) {
        double circumference = 2.0 * maxScore;
        double shifted = (score + maxScore + this.scoreShift) % circumference;
        if (shifted < 0.0) {
            shifted += circumference;
        }
        return shifted - maxScore;
    }

    /**
     * Sets the power, the offset and the rate of the reference density of the interval of the given node, which depend
     * only on what is the same before and after the move: the mutations, the rates of the genetic prior, and the
     * given means of the old and new values of the heights below the node. Returns false if the density is not proper.
     *
     * <p>The branch to a child with a gap g to the older child has the length x + g and the genetic prior ∝
     * (x + g)^m exp(-Λ (x + g)) for its m mutations and its rate Λ, which is the branch rate times the mean of the
     * total mutation rates at both of its ends. The two branches together are approximated by (x + s)^m exp(-Λ x) with
     * the total number of mutations m and the mean gap s of the mutations, so that the density is flat at the
     * lower end if the mutations sit on a branch with a large gap. The tree prior adds the slope of its approximation
     * at the typical height.
     */
    private boolean computeReference(Node node, double anchorHeight, double leftGap, double rightGap) {
        Node left = node.getLeft();
        Node right = node.getRight();

        int numLeftMutations = this.mutations.getMutations(left).size();
        int numRightMutations = this.mutations.getMutations(right).size();
        int numMutations = numLeftMutations + numRightMutations;
        double geneticRate = this.computeBranchRate(left, node) + this.computeBranchRate(right, node);

        double typicalInterval = (numMutations + 1.0) / Math.max(geneticRate, Double.MIN_NORMAL);
        if (Double.isInfinite(typicalInterval)) {
            return false;
        }

        this.power = numMutations;
        this.offset = numMutations == 0 ? 0.0 : (numLeftMutations * leftGap + numRightMutations * rightGap) / numMutations;

        this.treePriorHeight = anchorHeight + typicalInterval;
        this.rate = geneticRate;
        LearnedTreePrior treePrior = this.getTreePriorApproximation();
        if (treePrior != null) {
            double slope = treePrior.getSlope(this.treePriorHeight, this.getTreePriorParameters());
            this.rate = Math.max(geneticRate - MEAN_NUM_RAISED_NODES * slope, MIN_RATE_SHARE * geneticRate);
        }

        return this.rate > 0.0 && !Double.isInfinite(this.rate);
    }

    /** Returns the rate at which the mutations on the branch to the given child occur, which the move does not change. */
    private double computeBranchRate(Node child, Node parent) {
        double meanTotalMutationRate = (this.geneticPrior.getTotalMutationRate(child) + this.geneticPrior.getTotalMutationRate(parent)) / 2.0;
        return this.geneticPrior.branchRateModel.getRateForBranch(child) * meanTotalMutationRate;
    }

    /** Tunes the step size only if enough of the intervals moved, as otherwise the acceptance does not depend on it. */
    @Override
    protected boolean isStepTuned() {
        return this.numMovedIntervals >= MIN_MOVED_SHARE * this.numIntervals;
    }

    @Override
    protected double getMaxStepSize() {
        return MAX_STEP_SIZE;
    }

}
