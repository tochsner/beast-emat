package emat.operators.mdspr;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.util.Randomizer;

import java.util.Map;

/**
 * Picks the root X of the subtree to regraft with probability proportional to a selection
 * weight among all nodes where neither X nor its parent is the root. Subclasses define the
 * weight and may learn from the outcome of the proposals. A weight may only change between
 * proposals, in {@link #prepare()} and in the callbacks, so that the forward and the backward
 * pick probability of a proposal use the same weights, apart from the effect of the move on
 * which nodes can be picked and on their branch lengths.
 */
public abstract class SubtreeSelector {

    protected final Tree tree;

    // the selection weights of all nodes in the current state, which are zero for nodes that cannot be picked
    private final double[] weights;

    // how often each node was picked
    private final int[] numPicks;

    protected SubtreeSelector(Tree tree) {
        this.tree = tree;
        this.weights = new double[tree.getNodeCount()];
        this.numPicks = new int[tree.getNodeCount()];
    }

    /** Updates what the weights depend on. It is called once before every pick. */
    public void prepare() {
    }

    /** Picks X, or returns null without any candidate. */
    public Node pick() {
        double threshold = Randomizer.nextDouble() * this.computeWeights();

        Node lastCandidate = null;
        double cumulativeWeight = 0.0;
        for (Node node : this.tree.getNodesAsArray()) {
            double weight = this.weights[node.getNr()];
            if (weight <= 0.0) {
                continue;
            }

            lastCandidate = node;
            cumulativeWeight += weight;
            if (threshold < cumulativeWeight) {
                break;
            }
        }

        // without a break, the last candidate is only picked through rounding
        if (lastCandidate != null) {
            this.numPicks[lastCandidate.getNr()]++;
        }
        return lastCandidate;
    }

    /** Returns the log of the selection weight of X over the total weight of the candidates in the current state. */
    public double computeLogProbability(Node x) {
        double totalWeight = this.computeWeights();
        return Math.log(this.weights[x.getNr()] / totalWeight);
    }

    /** Learns from the accepted move of the picked subtree root. */
    public void acceptMove(Node picked, ProposedMove move) {
    }

    /** Learns from the rejected proposal for the picked subtree root. */
    public void rejectMove(Node picked) {
    }

    /** Returns the number of samples that the selection has estimated its weights from. */
    public int getNumSamples() {
        return 0;
    }

    /**
     * Adds the diagnostics of the picks so far: how evenly the subtrees are picked, and how
     * many candidates there are. The effective number of subtrees is (Σc)² / Σc² over the
     * pick counts c, or over the selection weights in the current state, and equals the
     * number of candidates if all are picked equally often.
     */
    public void addDiagnostics(Map<String, Double> diagnostics) {
        double totalPicks = 0.0;
        double totalSquaredPicks = 0.0;
        double tipPicks = 0.0;
        for (Node node : this.tree.getNodesAsArray()) {
            double picks = this.numPicks[node.getNr()];
            totalPicks += picks;
            totalSquaredPicks += picks * picks;
            if (node.isLeaf()) {
                tipPicks += picks;
            }
        }
        diagnostics.put("mdspr_effective_picked_subtrees", totalPicks * totalPicks / totalSquaredPicks);
        diagnostics.put("mdspr_tip_pick_fraction", tipPicks / totalPicks);

        double totalWeight = this.computeWeights();
        double totalSquaredWeight = 0.0;
        int numCandidates = 0;
        for (double weight : this.weights) {
            totalSquaredWeight += weight * weight;
            numCandidates += weight > 0.0 ? 1 : 0;
        }
        diagnostics.put("mdspr_effective_weighted_subtrees", totalWeight * totalWeight / totalSquaredWeight);
        diagnostics.put("mdspr_num_candidates", (double) numCandidates);
    }

    /** Computes the unnormalised probability of picking the given node as X. */
    protected abstract double computeWeight(Node node);

    /** Fills the selection weights of all nodes for the current state and returns their total. */
    private double computeWeights() {
        double totalWeight = 0.0;
        for (Node node : this.tree.getNodesAsArray()) {
            double weight = node.isRoot() || node.getParent().isRoot() ? 0.0 : this.computeWeight(node);
            this.weights[node.getNr()] = weight;
            totalWeight += weight;
        }
        return totalWeight;
    }
}
