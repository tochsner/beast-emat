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

        // the weights are indexed by node number, so the nodes are only looked up once picked
        int lastCandidateNr = -1;
        double cumulativeWeight = 0.0;
        for (int nodeNr = 0; nodeNr < this.weights.length; nodeNr++) {
            double weight = this.weights[nodeNr];
            if (weight <= 0.0) {
                continue;
            }

            lastCandidateNr = nodeNr;
            cumulativeWeight += weight;
            if (threshold < cumulativeWeight) {
                break;
            }
        }

        // without a break, the last candidate is only picked through rounding
        return lastCandidateNr < 0 ? null : this.tree.getNode(lastCandidateNr);
    }

    /** Returns the log of the selection weight of X over the total weight of the candidates in the current state. */
    public double computeLogProbability(Node x) {
        double totalWeight = this.computeWeights();
        return Math.log(this.weights[x.getNr()] / totalWeight);
    }

    /** Computes the unnormalised probability of picking the given node with the given parent as X. */
    protected abstract double computeWeight(Node node, Node parent);

    /** Fills the selection weights of all nodes for the current state and returns their total. */
    private double computeWeights() {
        double totalWeight = 0.0;
        for (Node node : this.tree.getNodesAsArray()) {
            Node parent = node.getParent();
            double weight = parent == null || parent.isRoot() ? 0.0 : this.computeWeight(node, parent);
            this.weights[node.getNr()] = weight;
            totalWeight += weight;
        }
        return totalWeight;
    }
}
