package emat.state;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.core.Log;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.datatype.DataType;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.Operator;
import beast.base.inference.StateNode;
import beast.base.util.Randomizer;

import java.io.PrintStream;
import java.util.*;

@Description("Stores the explicit mutations on the given tree.")
public class Mutations extends StateNode {

    final public Input<TreeInterface> treeInput = new Input<>("tree", "", Input.Validate.REQUIRED);
    final public Input<Alignment> alignmentInput = new Input<>("alignment", "", Input.Validate.REQUIRED);

    TreeInterface tree;
    Alignment alignment;

    int numStates;
    int numNodes;
    int numSites;

    int[] referenceSequence;

    private List<Mutation>[] mutationsAboveNode;
    private List<Mutation>[] storedMutationsAboveNode;

    // whether the mutations above a node changed since the mutations were last marked clean
    private boolean[] isBranchDirty;

    // whether the stored mutations hold a snapshot that the current proposal has not restored yet
    private boolean hasStoredMutations = false;

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

        this.setReferenceSequence();

        this.isBranchDirty = new boolean[this.numNodes];
        Arrays.fill(this.isBranchDirty, true);

        this.storedMutationsAboveNode = new List[this.numNodes];
        this.storeMutations();
    }

    /* State Management */

    /**
     * Applies the given list of mutations to the branch above the given node.
     */
    public void applyMutations(Node node, List<Mutation> mutations, Operator operator) {
        this.startEditing(operator);
        this.mutationsAboveNode[node.getNr()] = List.copyOf(mutations);
        this.isBranchDirty[node.getNr()] = true;

        // perform sanity checks from time to time
        if (Randomizer.nextDouble() < 0.01) {
            this.performSanityChecks(node.getNr());
        }
    }

    /**
     * Replaces the mutations of every branch without going through an operator. This is
     * meant for initialisation only, so the replaced mutations are also stored.
     */
    public void initialiseMutations(List<List<Mutation>> mutationsAboveNode) {
        if (mutationsAboveNode.size() != this.numNodes) {
            throw new IllegalArgumentException("Expected mutations for " + this.numNodes + " nodes.");
        }

        for (int nodeNr = 0; nodeNr < this.numNodes; nodeNr++) {
            this.mutationsAboveNode[nodeNr] = List.copyOf(mutationsAboveNode.get(nodeNr));
            this.performSanityChecks(nodeNr);
        }
        Arrays.fill(this.isBranchDirty, true);

        this.storeMutations();
    }
    
    /**
     * Rebases the reference sequence onto the root sequence, so that the root carries no
     * mutations anymore. This does not change the sequence of any node. The reference array
     * is updated in place, as other components hold on to it. This must not be called
     * during a proposal, as the stored mutations above the root would then refer to the
     * previous reference sequence.
     */
    public void setReferenceToRoot() {
        if (this.hasStoredMutations) {
            throw new IllegalStateException("Cannot change the reference sequence during a proposal.");
        }

        int rootNr = this.tree.getRoot().getNr();

        // apply the root mutations in order, so the last one at a site determines its root state
        for (Mutation mutation : this.mutationsAboveNode[rootNr]) {
            this.referenceSequence[mutation.site()] = mutation.newState();
        }

        this.mutationsAboveNode[rootNr] = List.of();
        this.storedMutationsAboveNode[rootNr] = List.of();
        this.isBranchDirty[rootNr] = true;
        this.setSomethingIsDirty(true);
    }

    public int[] getReferenceSequence() {
        return this.referenceSequence;
    }

    /**
     * Uses the sequence of the first tip as the reference. Ambiguous or missing states are
     * replaced by the first state they are compatible with, as missations are not yet
     * supported.
     */
    public void setReferenceSequence() {
        DataType dataType = this.alignment.getDataType();
        this.referenceSequence = new int[this.numSites];

        for (int site = 0; site < this.numSites; site++) {
            int code = this.alignment.getPattern(0, this.alignment.getPatternIndex(site));
            this.referenceSequence[site] = dataType.getStatesForCode(code)[0];
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

        // check that every mutation belongs to this branch and changes the state of its site

        for (Mutation mutation : mutations) {
            if (mutation.nodeNr() != nodeNr) {
                throw new RuntimeException("Mutation is annotated with a different branch.");
            }
            if (mutation.oldState() == mutation.newState()) {
                throw new RuntimeException("Mutation does not change the state of its site.");
            }
        }

        Node node = this.tree.getNode(nodeNr);

        if (node.isRoot()) {
            this.performRootSanityChecks(mutations);
        } else {
            this.performNonRootSanityChecks(node, mutations);
        }
    }

    /**
     * Performs the sanity checks for mutations above the root. These encode the difference
     * between the reference sequence and the root sequence, so their times are meaningless
     * and only the states are checked.
     */
    private void performRootSanityChecks(List<Mutation> mutations) {
        Map<Integer, Mutation> lastMutationAtSite = new HashMap<>();

        for (Mutation mutation : mutations) {
            Mutation lastMutation = lastMutationAtSite.put(mutation.site(), mutation);

            // the first mutation at a site starts from the reference sequence
            int previousState = lastMutation == null
                    ? this.referenceSequence[mutation.site()]
                    : lastMutation.newState();

            if (previousState != mutation.oldState()) {
                throw new RuntimeException("Site is not consistent among subsequent mutations.");
            }
        }
    }

    /**
     * Performs the sanity checks for mutations above a non-root node. This checks that all
     * mutations lie on the branch, are sorted, and are consistent wrt. states and times.
     */
    private void performNonRootSanityChecks(Node node, List<Mutation> mutations) {
        // evolution runs forwards in time from the parent to the node

        double branchStartHeight = node.getParent().getHeight();
        double branchEndHeight = node.getHeight();

        // check that every mutation lies on the branch

        for (Mutation mutation : mutations) {
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

    /**
     * Returns whether the mutations above the given node may have changed since the
     * mutations were last marked clean, i.e. since the end of the previous MCMC step.
     */
    public boolean isDirty(Node node) {
        return this.isBranchDirty[node.getNr()];
    }

    public TreeInterface getTree() {
        return this.tree;
    }

    public Alignment getAlignment() {
        return this.alignment;
    }

    /* StateNode methods */

    /**
     * Marks the mutations as clean or dirty. MCMC calls this with false after every step,
     * whether it was accepted or rejected, which ends the current proposal.
     */
    @Override
    public void setEverythingDirty(boolean isDirty) {
        this.setSomethingIsDirty(isDirty);
        Arrays.fill(this.isBranchDirty, isDirty);
        if (!isDirty) {
            this.hasStoredMutations = false;
        }
    }

    /**
     * Stores the mutations before a proposal changes them. The mutations take the tree as an
     * input, so BEAST also treats them as a calculation node downstream of the tree: after a
     * proposal that changes the tree, State calls store() and restore() on them a second
     * time. Only the first store() of a proposal takes a snapshot, as a later one would
     * overwrite it with the proposed mutations.
     */
    @Override
    protected void store() {
        if (!this.hasStoredMutations) {
            this.storeMutations();
            this.hasStoredMutations = true;
        }
    }

    /**
     * Restores the mutations from before the proposal. Only the first restore() of a
     * proposal swaps back, as a second one would reinstate the rejected mutations.
     */
    @Override
    public void restore() {
        if (!this.hasStoredMutations) {
            return;
        }

        List<Mutation>[] mutationsAboveNode = this.mutationsAboveNode;
        this.mutationsAboveNode = this.storedMutationsAboveNode;
        this.storedMutationsAboveNode = mutationsAboveNode;
        this.hasStoredMutations = false;
    }

    private void storeMutations() {
        System.arraycopy(this.mutationsAboveNode, 0, this.storedMutationsAboveNode, 0, this.numNodes);
    }

    /** Returns a deep copy of the mutations. */
    @Override
    public Mutations copy() {
        try {
            Mutations copy = (Mutations) this.clone();
            copy.mutationsAboveNode = this.mutationsAboveNode.clone();
            copy.storedMutationsAboveNode = this.storedMutationsAboveNode.clone();
            copy.isBranchDirty = this.isBranchDirty.clone();
            return copy;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }

    /** Assigns all values of this to the other mutations (other := this). */
    @Override
    public void assignTo(StateNode other) {
        Mutations target = (Mutations) other;
        target.setID(this.getID());
        target.index = this.index;
        target.copyValuesFrom(this);
    }

    /** Assigns all values of the other mutations to this (this := other). */
    @Override
    public void assignFrom(StateNode other) {
        Mutations source = (Mutations) other;
        this.setID(source.getID());
        this.copyValuesFrom(source);
    }

    /** Assigns only the mutations of the other mutations to this, e.g. when resuming a run. */
    @Override
    public void assignFromFragile(StateNode other) {
        Mutations source = (Mutations) other;
        System.arraycopy(source.mutationsAboveNode, 0, this.mutationsAboveNode, 0, this.numNodes);
        Arrays.fill(this.isBranchDirty, true);
        this.setSomethingIsDirty(false);
    }

    /** Copies the tree, the alignment and the mutations of the given source into this. */
    private void copyValuesFrom(Mutations source) {
        this.tree = source.tree;
        this.alignment = source.alignment;

        this.numStates = source.numStates;
        this.numNodes = source.numNodes;
        this.numSites = source.numSites;

        this.referenceSequence = source.referenceSequence;

        this.mutationsAboveNode = source.mutationsAboveNode.clone();
        this.storedMutationsAboveNode = source.storedMutationsAboveNode.clone();

        this.isBranchDirty = new boolean[this.numNodes];
        Arrays.fill(this.isBranchDirty, true);
    }

    /* Unsupported StateNode methods */

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
