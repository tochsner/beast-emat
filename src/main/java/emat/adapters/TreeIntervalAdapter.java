package emat.adapters;

import adapters.Adapter;
import beast.base.core.BEASTObject;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.Scalable;
import beast.base.inference.StateNode;
import emat.operators.nodetimes.BranchTimeMapper;
import emat.operators.nodetimes.NodeTimes;
import emat.state.Mutations;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Deque;
import java.util.List;

@Description("Adapter with a single mutable value, the logarithm of the sum of the intervals between every internal " +
        "node and its older child. Changing it scales all intervals by a common factor, keeps the ratios between " +
        "them and the relative positions of the mutations on their branches, and leaves the topology and the tips as they are.")
public class TreeIntervalAdapter extends BEASTObject implements Adapter {

    public final Input<Tree> treeInput = new Input<>("tree", "Tree whose intervals are scaled.", Input.Validate.REQUIRED);
    public final Input<Mutations> mutationsInput = new Input<>("mutations", "Mutations that move along with the branches.", Input.Validate.REQUIRED);

    public final Input<Scalable> downInput = new Input<>("down", "Parameter that is scaled inversely to the tree length, such as a clock rate, so that the product of the two stays the same along the line.");

    private Tree tree;
    private Mutations mutations;
    private Scalable down;

    // the nodes in an order that places every node after its children, fixed by refresh() for the duration of a proposal
    private Node[] internalNodesFromYoungest;
    private double[] baseIntervals;
    private double baseLogSum;
    private double baseTreeLength;
    private double appliedDownScale;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.mutations = this.mutationsInput.get();
        this.down = this.downInput.get();
        this.refresh();
    }

    /**
     * Stores the interval between every internal node and its older child. The map from the intervals to the node
     * heights is a bijection with Jacobian one, as every height is the height of its older child plus its interval,
     * so a scaling of the intervals is a line through the state that no state needs to be rejected on.
     */
    @Override
    public void refresh() {
        this.internalNodesFromYoungest = this.collectInternalNodesFromYoungest();
        this.baseIntervals = this.computeIntervals();
        this.baseLogSum = Math.log(Arrays.stream(this.baseIntervals).sum());
        this.baseTreeLength = NodeTimes.computeTreeLength(this.tree);
        this.appliedDownScale = 1.0;
    }

    /** Returns the internal nodes with every node placed after its children. */
    private Node[] collectInternalNodesFromYoungest() {
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

        Collections.reverse(nodes);
        return nodes.toArray(new Node[0]);
    }

    private double[] computeIntervals() {
        double[] intervals = new double[this.internalNodesFromYoungest.length];
        for (int i = 0; i < intervals.length; i++) {
            Node node = this.internalNodesFromYoungest[i];
            intervals[i] = node.getHeight() - Math.max(node.getLeft().getHeight(), node.getRight().getHeight());
        }
        return intervals;
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
    public double[] getMutable(int nodeId) {
        return new double[]{this.computeCurrentLogSum()};
    }

    /**
     * Scales the intervals stored by refresh() such that their sum has the given logarithm, places every node above
     * its higher child and maps the mutations along. A parameter to scale down is scaled by the inverse of the
     * change of the tree length.
     */
    @Override
    public double update(double[] mutable, int nodeId) {
        double factor = Math.exp(mutable[0] - this.baseLogSum);

        Node[] nodes = this.tree.getNodesAsArray();
        double[] oldHeights = new double[nodes.length];
        for (Node node : nodes) {
            oldHeights[node.getNr()] = node.getHeight();
        }

        for (int i = 0; i < this.internalNodesFromYoungest.length; i++) {
            Node node = this.internalNodesFromYoungest[i];
            node.setHeight(Math.max(node.getLeft().getHeight(), node.getRight().getHeight()) + factor * this.baseIntervals[i]);
        }

        BranchTimeMapper.mapMutations(this.mutations, nodes, oldHeights, null);

        if (this.down != null) {
            double downScale = this.baseTreeLength / NodeTimes.computeTreeLength(this.tree);
            this.down.scale(downScale / this.appliedDownScale);
            this.appliedDownScale = downScale;
        }

        return 0.0;
    }

    /**
     * Returns minus the log of the volume element of the tree and the mutation times along the scaling line: a
     * line through n intervals contributes a factor of the interval sum per interval, and the position of every
     * mutation on its branch contributes the length of the branch. A parameter that is scaled inversely to the
     * tree length contributes the inverse of the tree length, as the line follows the constant product of the two.
     * Slice sampling along the line then draws from the posterior density of the line coordinate.
     */
    @Override
    public double getLogJacobianCorrection(int nodeId) {
        double logVolume = this.internalNodesFromYoungest.length * this.computeCurrentLogSum();

        if (this.down != null) {
            logVolume -= Math.log(NodeTimes.computeTreeLength(this.tree));
        }

        for (Node node : this.tree.getNodesAsArray()) {
            if (!node.isRoot()) {
                logVolume += this.mutations.getMutations(node).size() * Math.log(node.getLength());
            }
        }

        return -logVolume;
    }

    private double computeCurrentLogSum() {
        return Math.log(Arrays.stream(this.computeIntervals()).sum());
    }

    @Override
    public List<StateNode> listStateNodes() {
        List<StateNode> stateNodes = new ArrayList<>(List.of(this.tree, this.mutations));
        if (this.down instanceof StateNode stateNode) {
            stateNodes.add(stateNode);
        }
        return stateNodes;
    }

}
