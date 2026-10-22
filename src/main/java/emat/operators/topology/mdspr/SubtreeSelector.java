package emat.operators.topology.mdspr;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.util.Randomizer;

/**
 * Picks the root X of the subtree to regraft with probability proportional to a selection
 * weight among all nodes where neither X nor its parent is the root. Subclasses define the
 * weight. The weights may only depend on the tree, so that the forward and the backward
 * pick probability of a proposal differ only by the effect of the move on the tree, such as
 * which nodes can be picked and their branch lengths.
 */
public abstract class SubtreeSelector {

    protected final Tree tree;

    // the selection weights of all nodes in the current state, which are zero for nodes that cannot be picked
    private final double[] weights;

    protected SubtreeSelector(Tree tree) {
        this.tree = tree;
        this.weights = new double[tree.getNodeCount()];
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
        return lastCandidate;
    }

    /** Returns the log of the selection weight of X over the total weight of the candidates in the current state. */
    public double computeLogProbability(Node x) {
        double totalWeight = this.computeWeights();
        return Math.log(this.weights[x.getNr()] / totalWeight);
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
