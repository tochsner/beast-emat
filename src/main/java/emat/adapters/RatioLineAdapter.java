package emat.adapters;

import adapters.Adapter;
import beast.base.core.BEASTObject;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.StateNode;
import emat.operators.nodetimes.BranchTimeMapper;
import emat.state.Mutations;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;

/**
 * Base of the adapters with a single mutable value that move the tree along a line in the ratio parameterisation of
 * the node heights. Every internal node below the root has the ratio r = (h - τ) / (h_p - τ) of its height h and
 * the height h_p of its parent, both above the height τ of the oldest tip below the node. For a fixed topology and
 * fixed tips, the height of the root above its oldest tip and these ratios determine all heights, and the map from
 * the ratios to the heights has the Jacobian Π (h_p - τ) over the internal nodes below the root. The mutations keep
 * their relative positions on their branches, which contributes the length of its branch once per mutation.
 */
public abstract class RatioLineAdapter extends BEASTObject implements Adapter {

    public final Input<Tree> treeInput = new Input<>("tree", "Tree whose heights are moved.", Input.Validate.REQUIRED);
    public final Input<Mutations> mutationsInput = new Input<>("mutations", "Mutations that move along with the branches.", Input.Validate.REQUIRED);

    protected Tree tree;
    protected Mutations mutations;

    // the internal nodes in pre-order, starting with the root, fixed by refresh() for the duration of a proposal
    protected Node[] internalNodes;

    // per node number, the height of the oldest tip below the node and the ratio at the start of the proposal
    protected double[] oldestTipHeights;
    protected double[] baseRatios;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.mutations = this.mutationsInput.get();
        this.refresh();
    }

    /** Stores the internal nodes, the height of the oldest tip below every node and the ratio of every internal node below the root. */
    @Override
    public void refresh() {
        this.internalNodes = this.collectInternalNodesFromRoot();
        this.oldestTipHeights = new double[this.tree.getNodeCount()];
        this.baseRatios = new double[this.tree.getNodeCount()];

        for (int i = this.internalNodes.length - 1; i >= 0; i--) {
            Node node = this.internalNodes[i];
            double oldestTipHeight = Double.NEGATIVE_INFINITY;
            for (Node child : node.getChildren()) {
                double childOldestTipHeight = child.isLeaf() ? child.getHeight() : this.oldestTipHeights[child.getNr()];
                oldestTipHeight = Math.max(oldestTipHeight, childOldestTipHeight);
            }
            this.oldestTipHeights[node.getNr()] = oldestTipHeight;
        }

        for (Node node : this.internalNodes) {
            if (!node.isRoot()) {
                double oldestTipHeight = this.oldestTipHeights[node.getNr()];
                this.baseRatios[node.getNr()] = (node.getHeight() - oldestTipHeight) / (node.getParent().getHeight() - oldestTipHeight);
            }
        }
    }

    /** Returns the internal nodes with every node placed before its children. */
    private Node[] collectInternalNodesFromRoot() {
        List<Node> nodes = new ArrayList<>();
        Deque<Node> stack = new ArrayDeque<>();
        stack.push(this.tree.getRoot());

        while (!stack.isEmpty()) {
            Node node = stack.pop();
            if (!node.isLeaf()) {
                nodes.add(node);
                stack.push(node.getLeft());
                stack.push(node.getRight());
            }
        }

        return nodes.toArray(new Node[0]);
    }

    /**
     * Sets the height of the root and then every other internal node to τ + r^power (h_p - τ) from its base ratio r
     * and the new height of its parent, and maps the mutations along. Throws if rounding puts a node at or above its parent.
     */
    protected void applyHeights(double rootHeight, double power) {
        Node[] nodes = this.tree.getNodesAsArray();
        double[] oldHeights = new double[nodes.length];
        for (Node node : nodes) {
            oldHeights[node.getNr()] = node.getHeight();
        }

        this.internalNodes[0].setHeight(rootHeight);
        for (int i = 1; i < this.internalNodes.length; i++) {
            Node node = this.internalNodes[i];
            double oldestTipHeight = this.oldestTipHeights[node.getNr()];
            double parentHeight = node.getParent().getHeight();
            double height = oldestTipHeight + Math.pow(this.baseRatios[node.getNr()], power) * (parentHeight - oldestTipHeight);
            if (!(height > oldestTipHeight && height < parentHeight)) {
                throw new IllegalStateException("a node is not strictly between its oldest tip and its parent");
            }
            node.setHeight(height);
        }

        BranchTimeMapper.mapMutations(this.mutations, nodes, oldHeights, null);
    }

    /**
     * Returns the log of the volume element of the ratios and the mutation times at the current state, without the
     * part along the line: the Jacobian Π (h_p - τ) of the map from the ratios to the heights over the internal nodes
     * below the root, and the length of every branch once per mutation on it.
     */
    protected double computeLogRatioVolume() {
        double logVolume = 0.0;

        for (Node node : this.tree.getNodesAsArray()) {
            if (node.isRoot()) {
                continue;
            }

            if (!node.isLeaf()) {
                logVolume += Math.log(node.getParent().getHeight() - this.oldestTipHeights[node.getNr()]);
            }
            logVolume += this.mutations.getMutations(node).size() * Math.log(node.getLength());
        }

        return logVolume;
    }

    @Override
    public int getNumImmutable() {
        return 0;
    }

    @Override
    public int getNumMutable() {
        return 1;
    }

    @Override
    public double[] getImmutable(int nodeId) {
        return new double[0];
    }

    @Override
    public List<StateNode> listStateNodes() {
        return List.of(this.tree, this.mutations);
    }

}
