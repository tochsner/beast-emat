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

        this.referenceSequence = new int[this.alignment.getSiteCount()];
        this.totalMutationRatesPerNode = new double[this.tree.getNodeCount()];

        if (this.m_siteModel.getCategoryCount() != 1) {
            throw new IllegalArgumentException("GeneticPrior does not support site rate heterogeneity.");
        }
    }

    @Override
    public double calculateLogP() {
        double logP = 0.0;

        // add root contributions

        double[] substitutionModelFrequencies = this.substitutionModel.getFrequencies();
        int[] stateOccurrences = this.mutations.getRootStateOccurrences();

        for (int stateNr = 0; stateNr < this.alignment.getMaxStateCount(); stateNr++) {
            logP += stateOccurrences[stateNr] * Math.log(substitutionModelFrequencies[stateNr]);
        }

        // add branch contributions

        for (int nodeNr = 0; nodeNr < this.tree.getNodeCount(); nodeNr++) {
            Node node = this.tree.getNode(nodeNr);
            if (node.isRoot()) {
                continue;
            }

            double branchLogP = this.calculateBranchContribution(node);
            logP += branchLogP;
        }

        this.logP = logP;
        return logP;
    }

    /**
     * Computes the likelihood contribution of a given branch.
     * There are two relevant factors at play: (i) the probability that no change
     * happens in-between the given mutations; (ii) the probability of the given
     * mutations.
     */
    private double calculateBranchContribution(Node node) {
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

    /** Returns the total mutation rate λ for the given node. */
    private double getTotalMutationRate(Node node) {
        if (node == null) {
            // we are asking for the reference sequence above the root
            return computeReferenceMutationRate();
        }

        if (this.totalMutationRatesPerNode[node.getNr()] == 0.0) {
            this.totalMutationRatesPerNode[node.getNr()] = this.computeTotalMutationRate(node);
        }

        return this.totalMutationRatesPerNode[node.getNr()];
    }

    /** Computes the total mutation rate λ for the reference sequence. */
    private double computeReferenceMutationRate() {
        double referenceMutationRate = 0.0;

        for (int i = 0; i < this.referenceSequence.length; i++) {
            referenceMutationRate += this.getEscapeRate(
                    this.tree.getRoot(), this.referenceSequence[i], i
            );
        }

        return referenceMutationRate;
    }

    /** Computes the total mutation rate λ for the given node. */
    private double computeTotalMutationRate(Node node) {
        Node parent = node.getParent();
        List<Mutation> mutations = this.mutations.getMutations(node);

        double totalMutationRate = this.getTotalMutationRate(parent);
        for (Mutation mutation : mutations) {
            totalMutationRate += this.getEscapeRate(node, mutation.newState(), mutation.site())
                    - this.getEscapeRate(node, mutation.oldState(), mutation.site());
        }

        return totalMutationRate;
    }

}
