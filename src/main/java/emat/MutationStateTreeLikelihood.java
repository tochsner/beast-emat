package emat;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.TreeInterface;
import beast.base.evolution.tree.Node;
import beast.base.spec.evolution.likelihood.TreeLikelihood;

@Description("Implements the genetic prior for EMATs. Missations and site-rate variation is not yet supported.")
public class MutationStateTreeLikelihood extends TreeLikelihood {

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "", Input.Validate.REQUIRED);

    TreeInterface tree;
    Mutations mutations;

    @Override
    public void initAndValidate() {
        super.initAndValidate();

        this.tree = this.treeInput.get();
        this.mutations = this.mutationsInput.get();
    }

    @Override
    public double calculateLogP() {
        double logP = 0.0;

        // add root contributions

        double[] substitutionModelFrequencies = this.substitutionModel.getFrequencies();
        int[] stateOccurrences = this.mutations.getRootStateOccurrences();

        for (int stateNr = 0; stateNr < this.mutations.getMaxStateCount(); stateNr++) {
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

        this.logP = logP
        return logP;
    }

    /**
     * Computes the likelihood contribution of a given branch.
     * There are two relevant factors at play: (i) the probability that no change
     * happens in-between the given mutations; (ii) the probability of the given
     * mutations.
     */
    private double calculateBranchContribution(Node node) {
        int numStates = this.mutations.getMaxStateCount();
        double[] rateMatrix = this.substitutionModel.getRateMatrix(node);
        double branchRate = this.branchRateModel.getRateForBranch(node);

        double branchLogP = 0.0;

        // add the contribution for no change between mutations

        double[] timeSpentPerState = this.mutations.getTimeSpentPerState(node);
        for (int stateNr = 0; stateNr < numStates; stateNr++) {
            double stayingRate = rateMatrix[stateNr*numStates + stateNr];
            branchLogP += stayingRate * branchRate * timeSpentPerState[stateNr];
        }

        // add the contribution of the explicit mutations

        int[][] numberOfMutations = this.mutations.getNumberOfMutations(node);
        for (int stateNr1 = 0; stateNr1 < numStates; stateNr1++) {
            for (int stateNr2 = 0; stateNr2 < numStates; stateNr2++) {
                if (stateNr1 == stateNr2) continue;

                double mutationRate = rateMatrix[stateNr1*numStates + stateNr2];
                branchLogP += Math.log(mutationRate * branchRate) * numberOfMutations[stateNr1][stateNr2];
            }
        }

        return branchLogP;
    }

}
