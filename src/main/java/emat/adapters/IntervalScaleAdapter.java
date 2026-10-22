package emat.adapters;

import adapters.Adapter;
import beast.base.core.BEASTObject;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.StateNode;
import emat.state.Mutation;
import emat.state.Mutations;

import java.util.ArrayList;
import java.util.List;

@Description("Mutable adapter with a single coordinate, the logarithm of the sum of the intervals between every " +
        "internal node and its older child. Updating it scales all of these intervals by a common factor and maps " +
        "every mutation linearly within its branch, like the IntervalScaleOperator. Unlike scaling all heights, " +
        "this keeps the tree and its mutations valid even for tips sampled through time.")
public class IntervalScaleAdapter extends BEASTObject implements Adapter {

    public final Input<Tree> treeInput = new Input<>("tree", "The tree to scale.", Input.Validate.REQUIRED);
    public final Input<Mutations> mutationsInput = new Input<>("mutations", "The mutations to move along with the tree.", Input.Validate.REQUIRED);

    private Tree tree;
    private Mutations mutations;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.mutations = this.mutationsInput.get();
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
        return new double[]{Math.log(this.computeIntervalSum(this.tree.getRoot()))};
    }

    /**
     * Sums the interval between every internal node and its older child below the given node.
     */
    private double computeIntervalSum(Node node) {
        if (node.isLeaf()) {
            return 0.0;
        }

        double interval = node.getHeight() - Math.max(node.getLeft().getHeight(), node.getRight().getHeight());
        return interval + this.computeIntervalSum(node.getLeft()) + this.computeIntervalSum(node.getRight());
    }

    /**
     * Scales the intervals so that their logarithmic sum becomes the given value, and moves every
     * mutation along with its branch. The log Jacobian of this map on the heights of the internal nodes
     * and the times of the mutations is the change of {@link #getLogJacobianCorrection}.
     */
    @Override
    public double update(double[] mutable, int nodeId) {
        double scaler = Math.exp(mutable[0]) / this.computeIntervalSum(this.tree.getRoot());

        // remember the old heights to map the mutations of every branch

        Node[] nodes = this.tree.getNodesAsArray();
        double[] oldHeights = new double[nodes.length];
        for (Node node : nodes) {
            oldHeights[node.getNr()] = node.getHeight();
        }

        this.scaleIntervals(this.tree.getRoot(), scaler);

        // move the mutations of every non-root branch along with its ends

        for (Node node : nodes) {
            List<Mutation> branchMutations = this.mutations.getMutations(node);
            if (node.isRoot() || branchMutations.isEmpty()) {
                // the times of the root mutations are meaningless
                continue;
            }

            double oldStartHeight = oldHeights[node.getParent().getNr()];
            double oldEndHeight = oldHeights[node.getNr()];
            double newStartHeight = node.getParent().getHeight();
            double newEndHeight = node.getHeight();

            List<Mutation> newBranchMutations = new ArrayList<>();
            for (Mutation mutation : branchMutations) {
                newBranchMutations.add(new Mutation(
                        mutation.nodeNr(),
                        this.mapHeight(mutation.time(), oldStartHeight, oldEndHeight, newStartHeight, newEndHeight),
                        this.mapHeight(mutation.timeOfPreviousMutation(), oldStartHeight, oldEndHeight, newStartHeight, newEndHeight),
                        mutation.site(), mutation.oldState(), mutation.newState()
                ));
            }

            this.mutations.applyMutations(node, newBranchMutations, null);
        }

        return 0.0;
    }

    /**
     * Scales the interval between the given node and its older child by the given factor,
     * after scaling everything below it. Returns the number of scaled nodes.
     */
    private int scaleIntervals(Node node, double scaler) {
        if (node.isLeaf()) {
            return 0;
        }

        double oldInterval = node.getHeight() - Math.max(node.getLeft().getHeight(), node.getRight().getHeight());

        int numScaledNodes = 1;
        numScaledNodes += this.scaleIntervals(node.getLeft(), scaler);
        numScaledNodes += this.scaleIntervals(node.getRight(), scaler);

        double minHeight = Math.max(node.getLeft().getHeight(), node.getRight().getHeight());
        node.setHeight(minHeight + oldInterval * scaler);

        return numScaledNodes;
    }

    /**
     * Maps a height on a branch linearly from its old to its new ends. The ends map exactly
     * onto the new ends, which keeps the times of previous mutations at the branch start
     * consistent, and rounding cannot push a height off the branch.
     */
    private double mapHeight(double height, double oldStartHeight, double oldEndHeight, double newStartHeight, double newEndHeight) {
        if (height == oldStartHeight) {
            return newStartHeight;
        }

        double fraction = (height - oldEndHeight) / (oldStartHeight - oldEndHeight);
        double newHeight = newEndHeight + fraction * (newStartHeight - newEndHeight);
        return Math.min(Math.max(newHeight, newEndHeight), newStartHeight);
    }

    /**
     * Returns minus the sum of the logarithms of all intervals and of the branch lengths, the latter counted once
     * for every mutation on the branch. Scaling the intervals by a common factor scales each of them and each
     * branch length with a mutation by a factor, so the correction before the update minus the correction after
     * it is the log Jacobian of the update.
     */
    @Override
    public double getLogJacobianCorrection(int nodeId) {
        double logSum = 0.0;

        for (Node node : this.tree.getNodesAsArray()) {
            if (!node.isLeaf()) {
                logSum += Math.log(node.getHeight() - Math.max(node.getLeft().getHeight(), node.getRight().getHeight()));
            }

            if (!node.isRoot()) {
                logSum += this.mutations.getMutations(node).size() * Math.log(node.getLength());
            }
        }

        return -logSum;
    }

    @Override
    public List<StateNode> listStateNodes() {
        return List.of(this.tree, this.mutations);
    }

}
