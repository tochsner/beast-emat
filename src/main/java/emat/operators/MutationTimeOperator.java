package emat.operators;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.Operator;
import beast.base.util.Randomizer;
import emat.state.Mutation;
import emat.state.Mutations;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

@Description("Moves a random mutation to a uniformly random time between its neighbouring events at the same site.")
public class MutationTimeOperator extends Operator {

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to operate on", Input.Validate.REQUIRED);

    Mutations mutations;
    TreeInterface tree;

    @Override
    public void initAndValidate() {
        this.mutations = this.mutationsInput.get();
        this.tree = this.mutations.getTree();
    }

    /**
     * Picks a mutation uniformly among all mutations on non-root branches and draws its new
     * time uniformly between the previous event at its site (the previous mutation or the
     * branch start) and the next event at its site (the next mutation or the branch end).
     * These bounds do not depend on the current time, and the number of mutations does not
     * change, so the proposal is symmetric.
     */
    @Override
    public double proposal() {
        int numMutations = this.countNonRootMutations();
        if (numMutations == 0) {
            return Double.NEGATIVE_INFINITY;
        }

        // find the branch and the index of the chosen mutation

        int mutationNr = Randomizer.nextInt(numMutations);

        Node node = null;
        for (Node candidate : this.tree.getNodesAsArray()) {
            if (candidate.isRoot()) {
                continue;
            }

            int branchMutationCount = this.mutations.getMutations(candidate).size();
            if (mutationNr < branchMutationCount) {
                node = candidate;
                break;
            }
            mutationNr -= branchMutationCount;
        }

        List<Mutation> branchMutations = this.mutations.getMutations(node);
        Mutation mutation = branchMutations.get(mutationNr);

        // the mutations are sorted by descending height, so the next event at the site comes later in the list

        int nextMutationNr = -1;
        for (int i = mutationNr + 1; i < branchMutations.size(); i++) {
            if (branchMutations.get(i).site() == mutation.site()) {
                nextMutationNr = i;
                break;
            }
        }

        double upperHeight = mutation.timeOfPreviousMutation();
        double lowerHeight = nextMutationNr >= 0
                ? branchMutations.get(nextMutationNr).time()
                : node.getHeight();

        double newTime = lowerHeight + Randomizer.nextDouble() * (upperHeight - lowerHeight);

        // move the mutation, and let the next mutation at the site start from the new time

        List<Mutation> newBranchMutations = new ArrayList<>(branchMutations);

        newBranchMutations.set(mutationNr, new Mutation(
                mutation.nodeNr(), newTime, mutation.timeOfPreviousMutation(),
                mutation.site(), mutation.oldState(), mutation.newState()
        ));

        if (nextMutationNr >= 0) {
            Mutation nextMutation = branchMutations.get(nextMutationNr);
            newBranchMutations.set(nextMutationNr, new Mutation(
                    nextMutation.nodeNr(), nextMutation.time(), newTime,
                    nextMutation.site(), nextMutation.oldState(), nextMutation.newState()
            ));
        }

        newBranchMutations.sort(Comparator.comparingDouble(Mutation::time).reversed());
        this.mutations.applyMutations(node, newBranchMutations, this);

        return 0.0;
    }

    /** Counts the mutations on all branches except the root branch, whose times are meaningless. */
    private int countNonRootMutations() {
        int numMutations = 0;
        for (Node node : this.tree.getNodesAsArray()) {
            if (!node.isRoot()) {
                numMutations += this.mutations.getMutations(node).size();
            }
        }
        return numMutations;
    }

}
