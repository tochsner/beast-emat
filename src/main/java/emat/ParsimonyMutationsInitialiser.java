package emat;

import beast.base.core.BEASTObject;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.datatype.DataType;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.StateNode;
import beast.base.inference.StateNodeInitialiser;
import beast.base.util.Randomizer;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

@Description("Initialises the mutations with a Fitch parsimony reconstruction on the current tree.")
public class ParsimonyMutationsInitialiser extends BEASTObject implements StateNodeInitialiser {

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to initialise", Input.Validate.REQUIRED);

    Mutations mutations;
    TreeInterface tree;
    Alignment alignment;
    DataType dataType;

    int numStates;
    int numPatterns;

    // the Fitch state sets per node and pattern, as bit masks over the states
    long[][] stateSets;

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
        this.dataType = this.alignment.getDataType();

        this.numStates = this.alignment.getMaxStateCount();
        this.numPatterns = this.alignment.getPatternCount();

        if (this.numStates > Long.SIZE) {
            throw new IllegalArgumentException("Parsimony initialisation supports at most " + Long.SIZE + " states.");
        }

        int numNodes = this.tree.getNodeCount();
        this.stateSets = new long[numNodes][this.numPatterns];
        this.nodeStates = new int[numNodes][this.numPatterns];

        // run Fitch: collect the state sets upwards, then choose the states downwards

        Node root = this.tree.getRoot();
        this.computeStateSets(root);
        this.chooseStates(root, this.getReferencePatternStates());

        this.mutations.initialiseMutations(this.createMutations());
    }

    @Override
    public void getInitialisedStateNodes(List<StateNode> stateNodes) {
        stateNodes.add(this.mutationsInput.get());
    }

    /* Fitch Parsimony */

    /**
     * Computes the Fitch state sets of the given node and of every node below it. A tip's
     * set contains all states compatible with its observed character; an inner node's set
     * is the intersection of its children's sets, or their union if they do not intersect.
     */
    private void computeStateSets(Node node) {
        long[] nodeStateSets = this.stateSets[node.getNr()];

        if (node.isLeaf()) {
            int taxonNr = this.alignment.getTaxonIndex(node.getID());
            if (taxonNr < 0) {
                throw new IllegalArgumentException("Tip " + node.getID() + " is not in the alignment.");
            }

            for (int patternNr = 0; patternNr < this.numPatterns; patternNr++) {
                int code = this.alignment.getPattern(taxonNr, patternNr);
                nodeStateSets[patternNr] = this.getStateSet(code);
            }

            return;
        }

        for (Node child : node.getChildren()) {
            this.computeStateSets(child);
        }

        for (int patternNr = 0; patternNr < this.numPatterns; patternNr++) {
            long intersection = ~0L;
            long union = 0L;

            for (Node child : node.getChildren()) {
                intersection &= this.stateSets[child.getNr()][patternNr];
                union |= this.stateSets[child.getNr()][patternNr];
            }

            nodeStateSets[patternNr] = intersection != 0L ? intersection : union;
        }
    }

    /**
     * Chooses the states of the given node and of every node below it. A node keeps the
     * state of its parent if its state set allows it, which avoids a mutation; otherwise it
     * takes the lowest state in its set. Ambiguous or missing tip characters are thereby
     * filled in with the parent state.
     */
    private void chooseStates(Node node, int[] parentStates) {
        long[] nodeStateSets = this.stateSets[node.getNr()];
        int[] nodeStates = this.nodeStates[node.getNr()];

        for (int patternNr = 0; patternNr < this.numPatterns; patternNr++) {
            long stateSet = nodeStateSets[patternNr];
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

    /** Returns the set of states compatible with the given character code, as a bit mask. */
    private long getStateSet(int code) {
        long stateSet = 0L;
        for (int state : this.dataType.getStatesForCode(code)) {
            stateSet |= 1L << state;
        }
        return stateSet;
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
