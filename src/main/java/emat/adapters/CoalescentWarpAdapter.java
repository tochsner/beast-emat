package emat.adapters;

import adapters.Adapter;
import beast.base.core.BEASTObject;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.StateNode;
import beast.base.spec.inference.parameter.RealScalarParam;
import emat.operators.CoalescentWarpOperator;
import emat.state.Mutations;

import java.util.List;

@Description("Mutable adapter with two coordinates, the logarithm of the population size and the growth rate of an " +
        "exponential growth coalescent. Updating them moves all internal nodes so that every interval keeps its " +
        "length in coalescent time, moves the mutations along with their branches and sets the clock rate so that " +
        "its product with the tree length stays the same, like the CoalescentWarpOperator.")
public class CoalescentWarpAdapter extends BEASTObject implements Adapter {

    public final Input<Tree> treeInput = new Input<>("tree", "The tree to warp.", Input.Validate.REQUIRED);
    public final Input<Mutations> mutationsInput = new Input<>("mutations", "The mutations to move along with the tree.", Input.Validate.REQUIRED);
    public final Input<StateNode> popSizeInput = new Input<>("popSize", "The population size at height zero of the exponential growth coalescent.", Input.Validate.REQUIRED);
    public final Input<StateNode> growthRateInput = new Input<>("growthRate", "The growth rate of the exponential growth coalescent.", Input.Validate.REQUIRED);
    public final Input<StateNode> clockRateInput = new Input<>("clockRate", "The clock rate, which follows the tree length.", Input.Validate.REQUIRED);

    private Tree tree;
    private Mutations mutations;
    private RealScalarParam<?> popSize;
    private RealScalarParam<?> growthRate;
    private RealScalarParam<?> clockRate;
    private CoalescentWarpOperator warper;

    // whether the last update could not map the tree, which makes the proposal invalid
    private boolean isInvalid;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.mutations = this.mutationsInput.get();
        this.popSize = (RealScalarParam<?>) this.popSizeInput.get();
        this.growthRate = (RealScalarParam<?>) this.growthRateInput.get();
        this.clockRate = (RealScalarParam<?>) this.clockRateInput.get();

        // the warper only provides the map, so its step sizes are never used
        this.warper = new CoalescentWarpOperator();
        this.warper.initByName("weight", 1.0, "tree", this.tree, "mutations", this.mutations,
                "popSize", this.popSize, "growthRate", this.growthRate, "clockRate", this.clockRate,
                "sdLogPopSize", 1.0, "sdGrowthRate", 1.0, "correlation", 0.0, "optimise", false);
    }

    @Override
    public int getNumImmutable() {
        return 0;
    }

    @Override
    public int getNumMutable() {
        return 2;
    }

    @Override
    public double[] getImmutable(int nodeId) {
        return new double[0];
    }

    /**
     * Returns the coordinates of the current state. A new proposal starts by reading them, so this is also where
     * the outcome of the previous proposal is forgotten.
     */
    @Override
    public double[] getMutable(int nodeId) {
        this.isInvalid = false;
        return new double[]{Math.log(this.popSize.get()), this.growthRate.get()};
    }

    /**
     * Warps the tree, the mutations and the clock rate to the given population size and growth rate. If the tree
     * cannot be mapped, the proposal is made invalid through {@link #getLogJacobianCorrection}.
     */
    @Override
    public double update(double[] mutable, int nodeId) {
        double logPopSizeShift = mutable[0] - Math.log(this.popSize.get());
        double growthRateShift = mutable[1] - this.growthRate.get();

        double logHastingsRatio = this.warper.warp(logPopSizeShift, growthRateShift);
        this.isInvalid = !Double.isFinite(logHastingsRatio);

        return 0.0;
    }

    /**
     * Returns minus the logarithm of the density of the state under the flat measure of the coordinates, up to a
     * constant. The warp is a deterministic map of the heights, the mutation times, the clock rate and the
     * population size, so the correction before the update minus the correction after it is the log Jacobian of
     * the update. It has four parts. The heights contribute the sum over the internal nodes of the log inverse
     * population size at the node. The clock rate follows the tree length, which contributes its logarithm. The
     * population size contributes minus its logarithm, because the step is flat in the logarithm. The mutations
     * contribute minus the logarithm of the branch length once for every mutation.
     */
    @Override
    public double getLogJacobianCorrection(int nodeId) {
        if (this.isInvalid) {
            return Double.POSITIVE_INFINITY;
        }

        double logPopSize = Math.log(this.popSize.get());
        double growth = this.growthRate.get();

        double logInversePopSizes = 0.0;
        double treeLength = 0.0;
        double logMutationBranchLengths = 0.0;

        for (Node node : this.tree.getNodesAsArray()) {
            if (!node.isLeaf()) {
                logInversePopSizes += growth * node.getHeight() - logPopSize;
            }

            if (!node.isRoot()) {
                treeLength += node.getLength();
                logMutationBranchLengths += this.mutations.getMutations(node).size() * Math.log(node.getLength());
            }
        }

        return logInversePopSizes + Math.log(treeLength) - logPopSize - logMutationBranchLengths;
    }

    @Override
    public List<StateNode> listStateNodes() {
        return List.of(this.tree, this.mutations, this.popSize, this.growthRate, this.clockRate);
    }

}
