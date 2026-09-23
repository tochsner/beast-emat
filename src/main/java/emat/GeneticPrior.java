package emat;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.substitutionmodel.EigenDecomposition;
import beast.base.evolution.substitutionmodel.SubstitutionModel;
import beast.base.evolution.tree.TreeInterface;
import beast.base.evolution.tree.Node;
import beast.base.spec.evolution.branchratemodel.Base;
import beast.base.spec.evolution.branchratemodel.StrictClockModel;
import beast.base.spec.evolution.likelihood.GenericTreeLikelihood;
import beast.base.spec.evolution.sitemodel.SiteModel;

import java.util.List;

@Description("Implements the genetic prior for EMATs. Missations and varying site rates are not yet supported.")
public class GeneticPrior extends GenericTreeLikelihood {

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "", Input.Validate.REQUIRED);

    TreeInterface tree;
    Alignment alignment;
    Mutations mutations;

    SiteModel.Base siteModel;
    SubstitutionModel substitutionModel;
    Base branchRateModel;

    int numStates;

    double[] rateMatrix;

    double[] totalMutationRatesPerNode;
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
        this.totalMutationRatesPerNode = new double[this.tree.getNodeCount()];
    }

    @Override
    public double calculateLogP() {
        this.updateRateMatrix();
        this.updateTotalMutationRates();

        double logP = 0.0;

       for (int nodeNr = 0; nodeNr < this.tree.getNodeCount(); nodeNr++) {
            Node node = this.tree.getNode(nodeNr);
            logP += this.calculateBranchContribution(node);
        }

        this.logP = logP;
        return logP;
    }

    /**
     * Computes the likelihood contribution of a given branch.
     */
    private double calculateBranchContribution(Node node) {
        if (node.isRoot()) {
            return this.calculateRootBranchContribution(node);
        } else {
            return this.calculateNonRootBranchContribution(node);
        }
    }

    /** Computes the genetic prior for the root branch. */
    private double calculateRootBranchContribution(Node root) {
        double branchLogP = 0.0;
        double[] substitutionModelFrequencies = this.substitutionModel.getFrequencies();

        // add reference state contribution

        for (int state : this.referenceSequence) {
            branchLogP += Math.log(substitutionModelFrequencies[state]);
        }

        // add root mutation contributions

        List<Mutation> mutations = this.mutations.getMutations(root);

        for (Mutation mutation : mutations) {
            branchLogP += Math.log(substitutionModelFrequencies[mutation.newState()])
                    - Math.log(substitutionModelFrequencies[mutation.oldState()]);
        }

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

        return branchLogP;
    }

    /**
     * Recomputes the rate matrix Q from the eigen decomposition of the substitution model,
     * as Q = V diag(λ) V^-1. The rate matrices returned by general substitution models like
     * GTR are unreliable, as their eigen decomposition overwrites the stored matrix in place.
     * The substitution model is shared by all branches, so the matrix of the root is used.
     */
    private void updateRateMatrix() {
        EigenDecomposition eigenDecomposition = this.substitutionModel.getEigenDecomposition(this.tree.getRoot());
        double[] eigenVectors = eigenDecomposition.getEigenVectors();
        double[] inverseEigenVectors = eigenDecomposition.getInverseEigenVectors();
        double[] eigenValues = eigenDecomposition.getEigenValues();

        if (this.rateMatrix == null) {
            this.rateMatrix = new double[this.numStates * this.numStates];
        }

        for (int from = 0; from < this.numStates; from++) {
            for (int to = 0; to < this.numStates; to++) {
                double rate = 0.0;
                for (int k = 0; k < this.numStates; k++) {
                    rate += eigenVectors[from * this.numStates + k] * eigenValues[k]
                            * inverseEigenVectors[k * this.numStates + to];
                }
                this.rateMatrix[from * this.numStates + to] = rate;
            }
        }
    }

    /** Returns the rate of leaving the state on the given branch and site. */
    private double getEscapeRate(Node node, int state, int site) {
        return -this.getMutationRate(node, state, state, site);
    }

    /** Returns the rate of mutation the state from and to the given state on the given branch and site. */
    private double getMutationRate(Node node, int from, int to, int site) {
        double substitutionRate = this.rateMatrix[from * this.numStates + to];

        // category 0 because we don't support site-rate variation yet
        double siteRate = this.siteModel.getRateForCategory(0, node);

        return substitutionRate * siteRate;
    }

    /**
     * Recomputes the total mutation rate λ of every node. The root starts from the rate of
     * the reference sequence above it.
     */
    private void updateTotalMutationRates() {
        Node root = this.tree.getRoot();

        double referenceMutationRate = 0.0;
        for (int i = 0; i < this.referenceSequence.length; i++) {
            referenceMutationRate += this.getEscapeRate(root, this.referenceSequence[i], i);
        }

        this.updateTotalMutationRates(root, referenceMutationRate);
    }

    /**
     * Recomputes the total mutation rate λ of the given node and of every node below it,
     * given the rate of its parent. The mutations above the node shift the rate by the
     * difference of the escape rates.
     */
    private void updateTotalMutationRates(Node node, double parentMutationRate) {
        double totalMutationRate = parentMutationRate;
        for (Mutation mutation : this.mutations.getMutations(node)) {
            totalMutationRate += this.getEscapeRate(node, mutation.newState(), mutation.site())
                    - this.getEscapeRate(node, mutation.oldState(), mutation.site());
        }

        this.totalMutationRatesPerNode[node.getNr()] = totalMutationRate;

        for (Node child : node.getChildren()) {
            this.updateTotalMutationRates(child, totalMutationRate);
        }
    }

    /** Returns the total mutation rate λ for the given node. */
    private double getTotalMutationRate(Node node) {
        return this.totalMutationRatesPerNode[node.getNr()];
    }

}
