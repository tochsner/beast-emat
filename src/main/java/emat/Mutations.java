package emat;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.StateNode;

import java.io.PrintStream;
import java.util.*;

@Description("Stores the explicit mutations on the given tree.")
public class Mutations extends StateNode {

    Input<TreeInterface> treeInput = new Input<>("tree", "", Input.Validate.REQUIRED);
    Input<Alignment> alignmentInput = new Input<>("alignment", "", Input.Validate.REQUIRED);

    TreeInterface tree;
    Alignment alignment;

    private List<Mutation>[] mutationsAboveNode;
    private double[][] timeSpentPerState;
    private int[][][] numberOfMutations;

    int numStates;
    int numNodes;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.alignment = this.alignmentInput.get();

        // init state objects

        this.numStates = this.getMaxStateCount();
        this.numNodes = this.tree.getNodeCount();

        this.mutationsAboveNode = new List[numNodes];
        for (int i = 0; i < numNodes; i++) {
            this.mutationsAboveNode[i] = new ArrayList<>();
        }

        this.timeSpentPerState = new double[numNodes][numStates];
        this.numberOfMutations = new int[numNodes][numStates][numStates];
    }

    /* State Management */

    /**
     * Replaces the mutations on the branch above the given node.
     */
    public void replaceMutationsAboveNode(int nodeNr, List<Mutation> mutations) {
        this.mutationsAboveNode[nodeNr] = mutations;

        // reset internal state

        Arrays.fill(this.timeSpentPerState[nodeNr], 0.0);

        for (int otherNodeNr = 0; otherNodeNr < numNodes; otherNodeNr++) {
            Arrays.fill(this.numberOfMutations[nodeNr][otherNodeNr], 0);
        }

        // update internal state

        for (Mutation mutation : mutations) {
            this.timeSpentPerState[nodeNr][mutation.oldState()] += mutation.time() - mutation.timeOfPreviousMutation();
            this.numberOfMutations[nodeNr][mutation.oldState()][mutation.newState()]++;
        }

        // perform sanity checks

        this.performSanityChecks(nodeNr);
    }

    /**
     * Performs a range of sanity checks for mutations above the given node. This checks
     * that multiple mutations for a given site are consistent wrt. the states and times, and
     * that all mutations actually happen on the branch.
     * */
    private void performSanityChecks(int nodeNr) {
        List<Mutation> mutations = this.mutationsAboveNode[nodeNr];

        if (mutations.isEmpty()) {
            // nothing to check
            return;
        }

        Node node = this.tree.getNode(nodeNr);
        Node parent = node.getParent();

        double start = node.getHeight();
        double end = parent.getHeight();

        // check that the first mutation is correct

        if (mutations.getFirst().timeOfPreviousMutation() != node.getHeight()) {
            throw new RuntimeException("First mutation on branch does not have the correct timeOfPreviousMutation");
        }

        // check that all times are on the branch

        for (Mutation mutation : mutations) {
            if (mutation.timeOfPreviousMutation() < start) {
                throw new RuntimeException("timeOfPreviousMutation is earlier than the branch.");
            }
            if (mutation.time() < start) {
                throw new RuntimeException("time is earlier than the branch.");
            }
            if (end < mutation.timeOfPreviousMutation()) {
                throw new RuntimeException("timeOfPreviousMutation is later than the branch.");
            }
            if (end < mutation.time()) {
                throw new RuntimeException("time is later than the branch.");
            }
        }

        // check that consecutive mutations match

        Map<Integer, Mutation> lastMutationAtSite = new HashMap<>();
        for (Mutation mutation : mutations) {
            Mutation lastMutation = lastMutationAtSite.get(mutation.site());
            lastMutationAtSite.put(mutation.site(), mutation);

            if (lastMutation == null) continue;

            // check that the state is consistent

            if (lastMutation.newState() != mutation.oldState()) {
                throw new RuntimeException("Site is not consistent among subsequent mutations.");
            }

            // check that the time is consistent

            if (lastMutation.time() != mutation.timeOfPreviousMutation()) {
                throw new RuntimeException("Time is not consistent among subsequent mutations.");
            }
        }
    }

    /* Getter */

    public int getMaxStateCount() {
        return this.alignment.getMaxStateCount();
    }

    public int[] getRootStateOccurrences() {
        return new int[] {};
    }

    public double[] getTimeSpentPerState(Node node) {
        return this.timeSpentPerState[node.getNr()];
    }

    public int[][] getNumberOfMutations(Node node) {
        return this.numberOfMutations[node.getNr()];
    }

    /* StateNode methods */

    @Override
    public void setEverythingDirty(boolean b) {

    }

    @Override
    protected void store() {

    }

    @Override
    public void restore() {

    }

    /* Unsupported StateNode methods */


    @Override
    public StateNode copy() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void assignTo(StateNode stateNode) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void assignFrom(StateNode stateNode) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void assignFromFragile(StateNode stateNode) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void fromXML(org.w3c.dom.Node node) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void init(PrintStream printStream) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void log(long l, PrintStream printStream) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void close(PrintStream printStream) {
        throw new UnsupportedOperationException();
    }
}
