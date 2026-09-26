package emat.prior;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.substitutionmodel.EigenDecomposition;
import beast.base.evolution.substitutionmodel.SubstitutionModel;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.CalculationNode;
import beast.base.spec.evolution.branchratemodel.Base;
import beast.base.spec.evolution.branchratemodel.StrictClockModel;
import beast.base.spec.evolution.likelihood.GenericTreeLikelihood;
import beast.base.spec.evolution.sitemodel.SiteModel;
import emat.state.Mutation;
import emat.state.Mutations;

import java.util.List;

@Description("Implements the genetic prior for EMATs. Missations and varying site rates are not yet supported.")
public class GeneticPrior extends GenericTreeLikelihood {

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "", Input.Validate.REQUIRED);

    TreeInterface tree;
    Alignment alignment;
    Mutations mutations;

    public SiteModel.Base siteModel;
    public SubstitutionModel substitutionModel;
    public Base branchRateModel;

    int numStates;

    double[] rateMatrix;
    double siteRate;

    // relative changes of a total mutation rate below this are rounding from summing along a different path
    private static final double RATE_TOLERANCE = 1e-12;

    // the total mutation rate λ at the end of each branch
    double[] totalMutationRatesPerNode;
    double[] storedTotalMutationRatesPerNode;

    // the log contribution of each branch to the genetic prior
    double[] branchLogPs;
    double[] storedBranchLogPs;

    // whether the caches hold the values of a previous calculation
    boolean isInitialised = false;

    int[] referenceSequence;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.alignment = this.dataInput.get();
        this.mutations = this.mutationsInput.get();

        // set up the evolutionary model

        if (!(this.siteModelInput.get() instanceof SiteModel.Base)) {
            throw new IllegalArgumentException("siteModel input should be of type SiteModel.Base");
        }
        this.siteModel = (SiteModel.Base) this.siteModelInput.get();
        this.siteModel.setDataType(this.alignment.getDataType());
        this.substitutionModel = this.siteModel.substModelInput.get();

        if (this.siteModel.getCategoryCount() != 1) {
            throw new IllegalArgumentException("GeneticPrior does not support site rate heterogeneity.");
        }

        if (this.branchRateModelInput.get() != null) {
            this.branchRateModel = this.branchRateModelInput.get();
        } else {
            this.branchRateModel = new StrictClockModel();
        }

        this.numStates = this.alignment.getMaxStateCount();

        // set up the mutation state

        this.referenceSequence = this.mutations.getReferenceSequence();

        int numNodes = this.tree.getNodeCount();
        this.totalMutationRatesPerNode = new double[numNodes];
        this.storedTotalMutationRatesPerNode = new double[numNodes];
        this.branchLogPs = new double[numNodes];
        this.storedBranchLogPs = new double[numNodes];
    }

    /**
     * Computes the genetic prior and caches it as logP. Only the branches whose contribution
     * may have changed since the previous call are recomputed: those whose node is dirty in
     * the tree, whose mutations are dirty, or whose parent's total mutation rate changed. An
     * SPR move therefore only touches the few branches around the pruned and the grafted
     * subtree. Everything is recomputed on the first call or if the evolutionary model changed.
     */
    @Override
    public double calculateLogP() {
        Node root = this.tree.getRoot();

        // fetch and cache rates

        this.rateMatrix = this.computeRateMatrix();
        // category 0 because we don't support site-rate variation yet
        this.siteRate = this.siteModel.getRateForCategory(0, root);

        // update the branch contributions if needed

        boolean isEverythingDirty = !this.isInitialised
                || this.siteModel.somethingIsDirty()
                || (this.substitutionModel instanceof CalculationNode calculationNode && calculationNode.somethingIsDirty())
                || this.branchRateModel.somethingIsDirty();

        this.updateBranches(root, false, isEverythingDirty);
        this.isInitialised = true;

        // sum in node order, so that rounding does not depend on which branches were updated

        this.logP = 0.0;
        for (double branchLogP : this.branchLogPs) {
            this.logP += branchLogP;
        }

        return this.logP;
    }

    /**
     * Recomputes the contribution and the total mutation rate of every dirty branch at or
     * below the given node. A branch is dirty if its node is dirty in the tree, if its
     * mutations are dirty, or if the total mutation rate of its parent changed.
     */
    private void updateBranches(Node node, boolean isParentRateChanged, boolean isEverythingDirty) {
        boolean isRateChanged = false;

        if (isEverythingDirty || isParentRateChanged || node.isDirty() != Tree.IS_CLEAN || this.mutations.isDirty(node)) {
            double oldRate = this.getTotalMutationRate(node);
            this.branchLogPs[node.getNr()] = this.calculateBranchContribution(node);

            // the same mutations summed along a different path only differ by rounding, which must not spread down the tree
            isRateChanged = isEverythingDirty
                    || Math.abs(this.getTotalMutationRate(node) - oldRate) > RATE_TOLERANCE * Math.abs(oldRate);
        }

        for (Node child : node.getChildrenMutable()) {
            this.updateBranches(child, isRateChanged, isEverythingDirty);
        }
    }

    /**
     * Computes the likelihood contribution of a given branch and records the total mutation
     * rate λ at its end.
     */
    private double calculateBranchContribution(Node node) {
        if (node.isRoot()) {
            return this.calculateRootBranchContribution(node);
        } else {
            return this.calculateNonRootBranchContribution(node);
        }
    }

    /**
     * Computes the genetic prior for the root branch. The root starts from the rate of the
     * reference sequence above it.
     */
    private double calculateRootBranchContribution(Node root) {
        double branchLogP = 0.0;
        double rate = 0.0;
        double[] substitutionModelFrequencies = this.substitutionModel.getFrequencies();

        // add reference state contribution

        for (int i = 0; i < this.referenceSequence.length; i++) {
            branchLogP += Math.log(substitutionModelFrequencies[this.referenceSequence[i]]);
            rate += this.getEscapeRate(root, this.referenceSequence[i], i);
        }

        // add root mutation contributions

        List<Mutation> mutations = this.mutations.getMutations(root);

        for (Mutation mutation : mutations) {
            branchLogP += Math.log(substitutionModelFrequencies[mutation.newState()])
                    - Math.log(substitutionModelFrequencies[mutation.oldState()]);
            rate += this.getEscapeRate(root, mutation.newState(), mutation.site())
                    - this.getEscapeRate(root, mutation.oldState(), mutation.site());
        }

        this.totalMutationRatesPerNode[root.getNr()] = rate;

        return branchLogP;
    }

    /** Computes the genetic prior for a non-root branch. */
    private double calculateNonRootBranchContribution(Node node) {
        Node parent = node.getParent();
        List<Mutation> mutations = this.mutations.getMutations(node);
        double branchRate = this.branchRateModel.getRateForBranch(node);

        double branchLogP = 0.0;

        // add the contribution of the explicit mutations

        for (Mutation mutation : mutations) {
            double mutationRate = this.getMutationRate(node, mutation.oldState(), mutation.newState(), mutation.site());
            branchLogP += Math.log(branchRate * mutationRate);
        }

        // add the contribution for no change between mutations

        double rate = this.getTotalMutationRate(parent);
        double previousHeight = parent.getHeight();

        for (Mutation mutation : mutations) {
            branchLogP -= branchRate * rate * (previousHeight - mutation.time());
            rate += this.getEscapeRate(node, mutation.newState(), mutation.site())
                    - this.getEscapeRate(node, mutation.oldState(), mutation.site());
            previousHeight = mutation.time();
        }

        branchLogP -= branchRate * rate * (previousHeight - node.getHeight());

        this.totalMutationRatesPerNode[node.getNr()] = rate;

        return branchLogP;
    }

    /**
     * Computes the rate matrix Q from the eigen decomposition of the substitution model,
     * as Q = V diag(λ) V^-1. The rate matrices returned by general substitution models like
     * GTR are unreliable, as their eigen decomposition overwrites the stored matrix in place.
     * The substitution model is shared by all branches, so the matrix of the root is used.
     */
    public double[] computeRateMatrix() {
        EigenDecomposition eigenDecomposition = this.substitutionModel.getEigenDecomposition(this.tree.getRoot());
        double[] eigenVectors = eigenDecomposition.getEigenVectors();
        double[] inverseEigenVectors = eigenDecomposition.getInverseEigenVectors();
        double[] eigenValues = eigenDecomposition.getEigenValues();

        double[] rateMatrix = new double[this.numStates * this.numStates];

        for (int from = 0; from < this.numStates; from++) {
            for (int to = 0; to < this.numStates; to++) {
                double rate = 0.0;
                for (int k = 0; k < this.numStates; k++) {
                    rate += eigenVectors[from * this.numStates + k] * eigenValues[k]
                            * inverseEigenVectors[k * this.numStates + to];
                }
                rateMatrix[from * this.numStates + to] = rate;
            }
        }

        return rateMatrix;
    }

    /** Returns the rate of leaving the state on the given branch and site. */
    private double getEscapeRate(Node node, int state, int site) {
        return -this.getMutationRate(node, state, state, site);
    }

    /** Returns the rate of mutation the state from and to the given state on the given branch and site. */
    private double getMutationRate(Node node, int from, int to, int site) {
        return this.rateMatrix[from * this.numStates + to] * this.siteRate;
    }

    /**
     * Returns the total mutation rate λ at the end of the branch above the given node, as of
     * the last calculation. It excludes the rate of the branch.
     */
    public double getTotalMutationRate(Node node) {
        return this.totalMutationRatesPerNode[node.getNr()];
    }

    /* CalculationNode methods */

    @Override
    public void store() {
        System.arraycopy(this.totalMutationRatesPerNode, 0, this.storedTotalMutationRatesPerNode, 0, this.totalMutationRatesPerNode.length);
        System.arraycopy(this.branchLogPs, 0, this.storedBranchLogPs, 0, this.branchLogPs.length);
        super.store();
    }

    @Override
    public void restore() {
        double[] totalMutationRatesPerNode = this.totalMutationRatesPerNode;
        this.totalMutationRatesPerNode = this.storedTotalMutationRatesPerNode;
        this.storedTotalMutationRatesPerNode = totalMutationRatesPerNode;

        double[] branchLogPs = this.branchLogPs;
        this.branchLogPs = this.storedBranchLogPs;
        this.storedBranchLogPs = branchLogPs;

        super.restore();
    }

}
