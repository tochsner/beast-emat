package emat;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.TreeInterface;
import beast.base.evolution.tree.Node;
import beast.base.spec.evolution.likelihood.TreeLikelihood;

import java.util.List;

@Description("Implements the genetic prior for EMATs. Missations and varying site rates are not yet supported.")
public class GeneticPrior extends TreeLikelihood {

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "", Input.Validate.REQUIRED);

    TreeInterface tree;
    Mutations mutations;

    double[] totalMutationRatesPerNode;
    int[] referenceSequence;

    @Override
    public void initAndValidate() {
        super.initAndValidate();

        this.tree = this.treeInput.get();
        this.mutations = this.mutationsInput.get();

        this.referenceSequence = this.mutations.getReferenceSequence();
        this.totalMutationRatesPerNode = new double[this.tree.getNodeCount()];

        if (this.m_siteModel.getCategoryCount() != 1) {
            throw new IllegalArgumentException("GeneticPrior does not support site rate heterogeneity.");
        }
    }

    @Override
    public double calculateLogP() {
        this.updateTotalMutationRates();

        double logP = 0.0;

       for (int nodeNr = 0; nodeNr < this.tree.getNodeCount(); nodeNr++) {
            Node node = this.tree.getNode(nodeNr);
            logP += this.calculateBranchContribution(node);
        }

        this.logP = logP;
        return logP;
    }

    @Override
    protected boolean requiresRecalculation() {
        // the super class does not know about the mutations, so they have to be checked here
        return this.mutations.somethingIsDirty() || super.requiresRecalculation();
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

    /** Returns the rate of leaving the state on the given branch and site. */
    private double getEscapeRate(Node node, int state, int site) {
        return -this.getMutationRate(node, state, state, site);
    }

    /** Returns the rate of mutation the state from and to the given state on the given branch and site. */
    private double getMutationRate(Node node, int from, int to, int site) {
        int numStates = this.alignment.getMaxStateCount();
        double[] rateMatrix = this.substitutionModel.getRateMatrix(node);
        double substitutionRate = rateMatrix[from*numStates + to];

        // category 0 because we don't support site-rate variation yet
        double siteRate = this.m_siteModel.getRateForCategory(0, node);

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
