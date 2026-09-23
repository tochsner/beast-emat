package emat;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.Operator;
import beast.base.inference.StateNode;

import java.io.PrintStream;
import java.util.*;

@Description("Stores the explicit mutations on the given tree.")
public class Mutations extends StateNode {

    Input<TreeInterface> treeInput = new Input<>("tree", "", Input.Validate.REQUIRED);
    Input<Alignment> alignmentInput = new Input<>("alignment", "", Input.Validate.REQUIRED);

    TreeInterface tree;
    Alignment alignment;

    int numStates;
    int numNodes;
    int numSites;

    private List<Mutation>[] mutationsAboveNode;
    private List<Mutation>[] storedMutationsAboveNode;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.alignment = this.alignmentInput.get();

        // init state objects

        this.numStates = this.alignment.getMaxStateCount();
        this.numNodes = this.tree.getNodeCount();
        this.numSites = this.alignment.getSiteCount();

        this.mutationsAboveNode = new List[this.numNodes];
        for (int i = 0; i < this.numNodes; i++) {
            this.mutationsAboveNode[i] = List.of();
        }

        this.storedMutationsAboveNode = new List[this.numNodes];
        this.store();
    }

    /* State Management */

    /**
     * Applies the given list of mutations to the branch above the given node.
     */
    public void applyMutations(Node node, List<Mutation> mutations, Operator operator) {
        this.startEditing(operator);
        this.mutationsAboveNode[node.getNr()] = List.copyOf(mutations);
        this.performSanityChecks(node.getNr());
    }

    public int[] getReferenceSequence() {
        return new int[this.alignment.getSiteCount()];
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

        if (node.isRoot()) {
            throw new RuntimeException("There is no branch above the root node that could carry mutations.");
        }

        // evolution runs forwards in time from the parent to the node

        double branchStartHeight = node.getParent().getHeight();
        double branchEndHeight = node.getHeight();

        // check that every mutation belongs to this branch and lies on it

        for (Mutation mutation : mutations) {
            if (mutation.nodeNr() != nodeNr) {
                throw new RuntimeException("Mutation is annotated with a different branch.");
            }
            if (mutation.oldState() == mutation.newState()) {
                throw new RuntimeException("Mutation does not change the state of its site.");
            }
            if (mutation.timeOfPreviousMutation() > branchStartHeight) {
                throw new RuntimeException("timeOfPreviousMutation is earlier than the branch.");
            }
            if (mutation.time() > branchStartHeight) {
                throw new RuntimeException("time is earlier than the branch.");
            }
            if (mutation.timeOfPreviousMutation() < branchEndHeight) {
                throw new RuntimeException("timeOfPreviousMutation is later than the branch.");
            }
            if (mutation.time() < branchEndHeight) {
                throw new RuntimeException("time is later than the branch.");
            }
            if (mutation.time() > mutation.timeOfPreviousMutation()) {
                throw new RuntimeException("Mutation happens before the previous mutation at its site.");
            }
        }

        // check that the mutations are sorted from the start of the branch to its end

        for (int i = 1; i < mutations.size(); i++) {
            if (mutations.get(i).time() > mutations.get(i - 1).time()) {
                throw new RuntimeException("Mutations are not sorted by descending height.");
            }
        }

        // check that consecutive mutations at the same site match

        Map<Integer, Mutation> lastMutationAtSite = new HashMap<>();
        for (Mutation mutation : mutations) {
            Mutation lastMutation = lastMutationAtSite.put(mutation.site(), mutation);

            if (lastMutation == null) {
                // the first mutation at a site has no predecessor, so it starts at the branch start
                if (mutation.timeOfPreviousMutation() != branchStartHeight) {
                    throw new RuntimeException("First mutation at a site does not start at the beginning of the branch.");
                }
                continue;
            }

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

    public List<Mutation> getMutations(Node node) {
        return this.mutationsAboveNode[node.getNr()];
    }

    /* StateNode methods */

    @Override
    public void setEverythingDirty(boolean isDirty) {
        this.setSomethingIsDirty(isDirty);
    }

    @Override
    protected void store() {
        System.arraycopy(this.mutationsAboveNode, 0, this.storedMutationsAboveNode, 0, this.numNodes);
    }

    @Override
    public void restore() {
        List<Mutation>[] mutationsAboveNode = this.mutationsAboveNode;
        this.mutationsAboveNode = this.storedMutationsAboveNode;
        this.storedMutationsAboveNode = mutationsAboveNode;
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
