package emat.initalisation;

import beast.base.core.BEASTObject;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.StateNode;
import beast.base.inference.StateNodeInitialiser;
import beast.base.util.Randomizer;
import emat.helper.FitchParsimony;
import emat.state.Mutation;
import emat.state.Mutations;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

@Description("Initialises the mutations with a Fitch parsimony reconstruction on the current tree.")
public class ParsimonyMutationsInitialiser extends BEASTObject implements StateNodeInitialiser {

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to initialise", Input.Validate.REQUIRED);

    Mutations mutations;
    TreeInterface tree;
    Alignment alignment;

    int numPatterns;

    // the Fitch state sets per node and pattern
    FitchParsimony fitchParsimony;

    // the reconstructed state per node and pattern
    int[][] nodeStates;

    @Override
    public void initAndValidate() {
        this.mutations = this.mutationsInput.get();
    }

    @Override
    public void initStateNodes() {
        this.tree = this.mutations.getTree();
        this.alignment = this.mutations.getAlignment();

        this.numPatterns = this.alignment.getPatternCount();

        this.nodeStates = new int[this.tree.getNodeCount()][this.numPatterns];

        // run Fitch: collect the state sets upwards, then choose the states downwards

        this.fitchParsimony = new FitchParsimony(this.alignment, this.tree);
        this.chooseStates(this.tree.getRoot(), this.getReferencePatternStates());

        this.mutations.initialiseMutations(this.createMutations());
        this.mutations.setReferenceToRoot();
    }

    @Override
    public void getInitialisedStateNodes(List<StateNode> stateNodes) {
        stateNodes.add(this.mutationsInput.get());
    }

    /* Fitch Parsimony */

    /**
     * Chooses the states of the given node and of every node below it. A node keeps the
     * state of its parent if its state set allows it, which avoids a mutation; otherwise it
     * takes the lowest state in its set. Ambiguous or missing tip characters are thereby
     * filled in with the parent state.
     */
    private void chooseStates(Node node, int[] parentStates) {
        int[] nodeStates = this.nodeStates[node.getNr()];

        for (int patternNr = 0; patternNr < this.numPatterns; patternNr++) {
            long stateSet = this.fitchParsimony.getStateSet(node, patternNr);
            int parentState = parentStates[patternNr];

            if ((stateSet & (1L << parentState)) != 0L) {
                nodeStates[patternNr] = parentState;
            } else {
                nodeStates[patternNr] = Long.numberOfTrailingZeros(stateSet);
            }
        }

        for (Node child : node.getChildren()) {
            this.chooseStates(child, nodeStates);
        }
    }

    /**
     * Returns the reference sequence per pattern. It serves as the parent state of the
     * root, so that the root only differs from the reference where parsimony requires it.
     */
    private int[] getReferencePatternStates() {
        int[] referenceSequence = this.mutations.getReferenceSequence();
        int[] referencePatternStates = new int[this.numPatterns];

        for (int site = 0; site < referenceSequence.length; site++) {
            referencePatternStates[this.alignment.getPatternIndex(site)] = referenceSequence[site];
        }

        return referencePatternStates;
    }

    /* Mutation Placement */

    /**
     * Creates the mutations of every branch from the reconstructed states. Every site
     * whose state differs from the parent state gets a single mutation. On the root branch
     * the times are meaningless, so they are set to the root height.
     */
    private List<List<Mutation>> createMutations() {
        int[] referencePatternStates = this.getReferencePatternStates();

        List<List<Mutation>> mutationsAboveNode = new ArrayList<>();
        for (int nodeNr = 0; nodeNr < this.tree.getNodeCount(); nodeNr++) {
            Node node = this.tree.getNode(nodeNr);

            if (node.isRoot()) {
                mutationsAboveNode.add(this.createBranchMutations(node, referencePatternStates));
            } else {
                mutationsAboveNode.add(this.createBranchMutations(node, this.nodeStates[node.getParent().getNr()]));
            }
        }

        return mutationsAboveNode;
    }

    /**
     * Creates the mutations on the branch above the given node. Each mutation is placed at
     * a uniformly random height on the branch, and the mutations are sorted from the start
     * of the branch to its end.
     */
    private List<Mutation> createBranchMutations(Node node, int[] parentStates) {
        int nodeNr = node.getNr();
        int[] nodeStates = this.nodeStates[nodeNr];

        // evolution runs forwards in time from the parent to the node

        double branchEndHeight = node.getHeight();
        double branchStartHeight = node.isRoot() ? branchEndHeight : node.getParent().getHeight();
        double branchLength = branchStartHeight - branchEndHeight;

        List<Mutation> mutations = new ArrayList<>();
        for (int site = 0; site < this.alignment.getSiteCount(); site++) {
            int patternNr = this.alignment.getPatternIndex(site);
            int oldState = parentStates[patternNr];
            int newState = nodeStates[patternNr];

            if (oldState == newState) {
                continue;
            }

            double time = branchEndHeight + Randomizer.nextDouble() * branchLength;
            mutations.add(new Mutation(nodeNr, time, branchStartHeight, site, oldState, newState));
        }

        mutations.sort(Comparator.comparingDouble(Mutation::time).reversed());
        return mutations;
    }

}
