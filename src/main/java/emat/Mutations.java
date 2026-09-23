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

    int numStates;
    int numNodes;
    int numSites;

    private List<Mutation>[] mutationsAboveNode;

    private int[][] stateOccurrences;

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
            this.mutationsAboveNode[i] = new ArrayList<>();
        }

        this.stateOccurrences = new int[this.numNodes][this.numStates];
    }

    /* State Management */

    /**
     * Recomputes the state occurrences, the time spent per state and the number of
     * mutations of every branch.
     */
    private void updateInternalState() {
        Node root = this.tree.getRoot();

        // walk downwards, so that a branch is always visited after its parent

        Deque<Node> nodesToVisit = new ArrayDeque<>();
        nodesToVisit.push(root);

        while (!nodesToVisit.isEmpty()) {
            Node node = nodesToVisit.pop();

            for (Node child : node.getChildren()) {
                this.updateBranchAboveNode(child);
                nodesToVisit.push(child);
            }
        }
    }

    /**
     * Recomputes the state occurrences at the given node, together with the time spent per
     * state and the number of mutations on the branch above it. Assumes that the state
     * occurrences at the parent node are already up to date.
     */
    private void updateBranchAboveNode(Node node) {
        int nodeNr = node.getNr();
        int parentNr = node.getParent().getNr();

        // evolution runs forwards in time from the parent to the node

        double branchStartHeight = node.getParent().getHeight();
        double branchEndHeight = node.getHeight();
        double branchLength = branchStartHeight - branchEndHeight;

        int[] parentStateOccurrences = this.stateOccurrences[parentNr];
        int[] nodeStateOccurrences = this.stateOccurrences[nodeNr];

        // start from every site spending the whole branch in the state it has at the parent

        for (int stateNr = 0; stateNr < this.numStates; stateNr++) {
            nodeStateOccurrences[stateNr] = parentStateOccurrences[stateNr];
        }

        // move each mutated site into its new state for the rest of the branch

        for (Mutation mutation : this.mutationsAboveNode[nodeNr]) {
            nodeStateOccurrences[mutation.oldState()]--;
            nodeStateOccurrences[mutation.newState()]++;
        }
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

    public int[] getRootStateOccurrences() {
        return this.getStateOccurrences(this.tree.getRoot());
    }

    public int[] getStateOccurrences(Node node) {
        return this.stateOccurrences[node.getNr()];
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
