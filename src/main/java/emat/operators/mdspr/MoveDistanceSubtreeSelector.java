package emat.operators.mdspr;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;

/**
 * Picks candidates in proportion to the running mean distance of their moves, so that
 * subtrees that move far are picked more often. The mean starts with a pseudocount of moves
 * at the mean distance of all recorded moves. Without any distance yet, every candidate gets
 * the same weight.
 */
public class MoveDistanceSubtreeSelector extends SubtreeSelector {

    /** The measure of the distance between the old and the new attachment point. */
    public enum Measure {
        NUM_NODES, PATH_LENGTH, SQUARED_PATH_LENGTH;

        double measure(ProposedMove move) {
            return switch (this) {
                case NUM_NODES -> move.numNodes();
                case PATH_LENGTH -> move.pathLength();
                case SQUARED_PATH_LENGTH -> move.pathLength() * move.pathLength();
            };
        }
    }

    private final Measure measure;
    private final double pseudocount;
    private final boolean countRejectedMoves;

    // per node, the number and the total distance of its recorded moves, and the same over all nodes
    private final int[] numMoves;
    private final double[] totalDistances;
    private long numAllMoves;
    private double totalAllDistance;

    /**
     * @param pseudocount        the number of pseudo-moves with the mean distance of all recorded moves that the mean of every node starts with
     * @param countRejectedMoves whether rejected proposals count as moves of distance zero, which makes the mean the expected distance per proposal
     */
    public MoveDistanceSubtreeSelector(Tree tree, Measure measure, double pseudocount, boolean countRejectedMoves) {
        super(tree);
        if (pseudocount <= 0.0) {
            throw new IllegalArgumentException("moveDistancePseudocount must be positive.");
        }

        this.measure = measure;
        this.pseudocount = pseudocount;
        this.countRejectedMoves = countRejectedMoves;
        this.numMoves = new int[tree.getNodeCount()];
        this.totalDistances = new double[tree.getNodeCount()];
    }

    @Override
    protected double computeWeight(Node node) {
        if (this.totalAllDistance <= 0.0) {
            return 1.0;
        }

        double priorMeanDistance = this.totalAllDistance / this.numAllMoves;
        int nr = node.getNr();
        return (this.pseudocount * priorMeanDistance + this.totalDistances[nr]) / (this.pseudocount + this.numMoves[nr]);
    }

    @Override
    public void acceptMove(Node picked, ProposedMove move) {
        this.recordMove(picked, this.measure.measure(move));
    }

    @Override
    public void rejectMove(Node picked) {
        if (this.countRejectedMoves) {
            this.recordMove(picked, 0.0);
        }
    }

    private void recordMove(Node picked, double distance) {
        int nr = picked.getNr();
        this.numMoves[nr]++;
        this.totalDistances[nr] += distance;
        this.numAllMoves++;
        this.totalAllDistance += distance;
    }
}
