package esat.state;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.datatype.DataType;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.Operator;
import beast.base.inference.StateNode;
import beast.base.util.Randomizer;

import java.io.PrintStream;
import java.util.Arrays;
import java.util.List;

/**
 * Stores the explicit state of every node of a tree. The reference sequence is the root
 * sequence, and every other node stores the sites at which it differs from the root. The
 * transition counts of every branch, i.e. the number of sites in each pair of states at the
 * parent and the node, are kept up to date as the sufficient statistics of the branch.
 * <p>
 * The changes are immutable and the counts are copied on write: the stored state shares all
 * arrays with the current state, and an array is only copied before its first change in a
 * proposal. Storing the state therefore takes time linear in the number of nodes, independent
 * of the number of states and changes.
 */
@Description("Stores the explicit states of every node of the given tree as changes compared to the root sequence, " +
        "together with the transition counts of every branch.")
public class NodeStates extends StateNode {

    final public Input<TreeInterface> treeInput = new Input<>("tree", "", Input.Validate.REQUIRED);
    final public Input<Alignment> alignmentInput = new Input<>("alignment", "", Input.Validate.REQUIRED);

    // the probability with which a change of the state is checked against a computation from scratch
    private static final double SANITY_CHECK_PROBABILITY = 0.01;

    TreeInterface tree;
    Alignment alignment;

    int numStates;
    int numNodes;
    int numSites;

    // the root sequence, to which the changes of every node refer
    private int[] referenceSequence;
    private int[] storedReferenceSequence;

    // per node, the sites at which the node differs from the root sequence, which is empty for the root
    private StateChanges[] changesAtNode;
    private StateChanges[] storedChangesAtNode;

    // per node, the number of sites with each pair of states at the parent and the node, row-major by the parent state
    private int[][] transitionCounts;
    private int[][] storedTransitionCounts;

    // the number of sites of the root sequence in each state
    private int[] rootStateCounts;
    private int[] storedRootStateCounts;

    // whether the states at a node or the counts above it changed since the states were last marked clean
    private boolean[] isBranchDirty;

    // the nodes whose branch is dirty, which is only complete if not every branch is dirty
    private int[] dirtyNodeNrs;
    private int numDirtyNodes;
    private boolean isEveryBranchDirty;

    // whether the stored state holds a snapshot that the current proposal has not restored yet
    private boolean hasStoredStates = false;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.alignment = this.alignmentInput.get();

        this.numStates = this.alignment.getMaxStateCount();
        this.numNodes = this.tree.getNodeCount();
        this.numSites = this.alignment.getSiteCount();

        // start with every node in the state of the first tip

        this.referenceSequence = this.computeFirstTipSequence();

        this.changesAtNode = new StateChanges[this.numNodes];
        for (int nodeNr = 0; nodeNr < this.numNodes; nodeNr++) {
            this.changesAtNode[nodeNr] = StateChanges.empty(nodeNr);
        }

        this.transitionCounts = new int[this.numNodes][];
        this.recomputeAllCounts();

        this.isBranchDirty = new boolean[this.numNodes];
        this.dirtyNodeNrs = new int[this.numNodes];
        this.markEveryBranchDirty();

        this.storedChangesAtNode = new StateChanges[this.numNodes];
        this.storedTransitionCounts = new int[this.numNodes][];
        this.storeStates();
    }

    /**
     * Returns the sequence of the first tip. Ambiguous or missing states are replaced by the
     * first state they are compatible with.
     */
    private int[] computeFirstTipSequence() {
        DataType dataType = this.alignment.getDataType();
        int[] sequence = new int[this.numSites];

        for (int site = 0; site < this.numSites; site++) {
            int code = this.alignment.getPattern(0, this.alignment.getPatternIndex(site));
            sequence[site] = dataType.getStatesForCode(code)[0];
        }

        return sequence;
    }

    /** Recomputes the root state counts and the transition counts of every branch from the changes. */
    private void recomputeAllCounts() {
        this.rootStateCounts = this.computeRootStateCounts();

        for (Node node : this.tree.getNodesAsArray()) {
            this.transitionCounts[node.getNr()] = node.isRoot()
                    ? new int[this.numStates * this.numStates]
                    : this.computeTransitionCounts(node);
        }
    }

    private int[] computeRootStateCounts() {
        int[] counts = new int[this.numStates];
        for (int state : this.referenceSequence) {
            counts[state]++;
        }
        return counts;
    }

    /**
     * Computes the transition counts of the branch above the given non-root node by merging the
     * changes of its parent and of the node. Every site starts on the diagonal in its root state,
     * and only the sites listed by either of the two nodes are moved.
     */
    private int[] computeTransitionCounts(Node node) {
        int[] counts = new int[this.numStates * this.numStates];
        for (int state = 0; state < this.numStates; state++) {
            counts[state * this.numStates + state] = this.rootStateCounts[state];
        }

        StateChanges parentChanges = this.changesAtNode[node.getParent().getNr()];
        StateChanges nodeChanges = this.changesAtNode[node.getNr()];

        int i = 0;
        int j = 0;
        while (i < parentChanges.size() || j < nodeChanges.size()) {
            int parentSite = i < parentChanges.size() ? parentChanges.getSite(i) : Integer.MAX_VALUE;
            int nodeSite = j < nodeChanges.size() ? nodeChanges.getSite(j) : Integer.MAX_VALUE;
            int site = Math.min(parentSite, nodeSite);

            int rootState = this.referenceSequence[site];
            int parentState = parentSite == site ? parentChanges.getState(i++) : rootState;
            int nodeState = nodeSite == site ? nodeChanges.getState(j++) : rootState;

            counts[rootState * this.numStates + rootState]--;
            counts[parentState * this.numStates + nodeState]++;
        }

        return counts;
    }

    /* State Management */

    /**
     * Replaces the changes of the given non-root node. The transition counts of the branch
     * above the node and of the branches above its children are updated incrementally, which
     * only visits the sites whose state at the node actually changes.
     */
    public void applyChanges(Node node, StateChanges changes, Operator operator) {
        if (node.isRoot()) {
            throw new IllegalArgumentException("The root has no changes, its states are changed by applyRootStates.");
        }
        if (changes.getNodeNr() != node.getNr()) {
            throw new IllegalArgumentException("Changes are annotated with a different node.");
        }

        this.startEditing(operator);

        this.updateCountsOfChangedSites(node, this.changesAtNode[node.getNr()], changes);
        this.changesAtNode[node.getNr()] = changes;

        this.markBranchDirty(node.getNr());
        for (Node child : node.getChildrenMutable()) {
            this.markBranchDirty(child.getNr());
        }

        // perform sanity checks from time to time
        if (Randomizer.nextDouble() < SANITY_CHECK_PROBABILITY) {
            this.performSanityChecks(node);
        }
    }

    /**
     * Updates the transition counts for every site at which the state of the given node differs
     * between the old and the new changes, by merging the two.
     */
    private void updateCountsOfChangedSites(Node node, StateChanges oldChanges, StateChanges newChanges) {
        int i = 0;
        int j = 0;
        while (i < oldChanges.size() || j < newChanges.size()) {
            int oldSite = i < oldChanges.size() ? oldChanges.getSite(i) : Integer.MAX_VALUE;
            int newSite = j < newChanges.size() ? newChanges.getSite(j) : Integer.MAX_VALUE;
            int site = Math.min(oldSite, newSite);

            int rootState = this.referenceSequence[site];
            int oldState = oldSite == site ? oldChanges.getState(i++) : rootState;
            int newState = newSite == site ? newChanges.getState(j++) : rootState;

            if (oldState != newState) {
                this.moveSiteState(node, site, oldState, newState);
            }
        }
    }

    /**
     * Updates the transition counts of the branch above the given node and of the branches
     * above its children for a change of the state of the node at the given site.
     */
    private void moveSiteState(Node node, int site, int oldState, int newState) {
        int parentState = this.getState(node.getParent(), site);
        int[] counts = this.getEditableCounts(node.getNr());
        counts[parentState * this.numStates + oldState]--;
        counts[parentState * this.numStates + newState]++;

        for (Node child : node.getChildrenMutable()) {
            int childState = this.getState(child, site);
            int[] childCounts = this.getEditableCounts(child.getNr());
            childCounts[oldState * this.numStates + childState]--;
            childCounts[newState * this.numStates + childState]++;
        }
    }

    /**
     * Sets the states of the root at the given sites, given as two parallel arrays in any order.
     * Every other node keeps its states, so its changes are rewritten at the sites whose root
     * state changes. The transition counts only change for the branches above the children of
     * the root.
     */
    public void applyRootStates(int[] sites, int[] states, Operator operator) {
        Node root = this.tree.getRoot();
        StateChanges rootStates = StateChanges.of(root.getNr(), sites, states);

        // keep the sites whose root state changes, in ascending order

        int[] changedSites = new int[rootStates.size()];
        int[] newRootStates = new int[rootStates.size()];
        int numChangedSites = 0;
        for (int i = 0; i < rootStates.size(); i++) {
            int site = rootStates.getSite(i);
            if (rootStates.getState(i) != this.referenceSequence[site]) {
                changedSites[numChangedSites] = site;
                newRootStates[numChangedSites] = rootStates.getState(i);
                numChangedSites++;
            }
        }

        if (numChangedSites == 0) {
            return;
        }

        this.startEditing(operator);

        // update the counts of the root and of the branches above its children

        int[] rootStateCounts = this.getEditableRootStateCounts();
        for (int i = 0; i < numChangedSites; i++) {
            rootStateCounts[this.referenceSequence[changedSites[i]]]--;
            rootStateCounts[newRootStates[i]]++;
        }

        for (Node child : root.getChildrenMutable()) {
            int[] counts = this.getEditableCounts(child.getNr());
            for (int i = 0; i < numChangedSites; i++) {
                int site = changedSites[i];
                int childState = this.getState(child, site);
                counts[this.referenceSequence[site] * this.numStates + childState]--;
                counts[newRootStates[i] * this.numStates + childState]++;
            }
        }

        // rewrite the changes of every other node relative to the new root states

        for (Node node : this.tree.getNodesAsArray()) {
            if (!node.isRoot()) {
                this.changesAtNode[node.getNr()] = this.rebaseChanges(
                        this.changesAtNode[node.getNr()], changedSites, newRootStates, numChangedSites
                );
            }
        }

        int[] referenceSequence = this.getEditableReferenceSequence();
        for (int i = 0; i < numChangedSites; i++) {
            referenceSequence[changedSites[i]] = newRootStates[i];
        }

        this.markBranchDirty(root.getNr());
        for (Node child : root.getChildrenMutable()) {
            this.markBranchDirty(child.getNr());
        }

        // perform sanity checks from time to time
        if (Randomizer.nextDouble() < SANITY_CHECK_PROBABILITY) {
            this.performSanityChecks(root);
        }
    }

    /**
     * Returns the given changes rewritten for new root states at the given ascending sites,
     * keeping the states of the node. A site is listed afterwards if and only if the node
     * differs from its new root state. This must be called before the reference sequence is
     * updated, as unlisted sites take their old root state from it.
     */
    private StateChanges rebaseChanges(StateChanges changes, int[] changedSites, int[] newRootStates, int numChangedSites) {
        int maxSize = changes.size() + numChangedSites;
        int[] sites = new int[maxSize];
        int[] states = new int[maxSize];
        int size = 0;

        int i = 0;
        int j = 0;
        while (i < changes.size() || j < numChangedSites) {
            int changeSite = i < changes.size() ? changes.getSite(i) : Integer.MAX_VALUE;
            int rootSite = j < numChangedSites ? changedSites[j] : Integer.MAX_VALUE;

            if (changeSite < rootSite) {
                // the root state of this site stays the same
                sites[size] = changeSite;
                states[size] = changes.getState(i++);
                size++;
                continue;
            }

            int nodeState = changeSite == rootSite ? changes.getState(i++) : this.referenceSequence[rootSite];
            if (nodeState != newRootStates[j]) {
                sites[size] = rootSite;
                states[size] = nodeState;
                size++;
            }
            j++;
        }

        return StateChanges.ofSorted(changes.getNodeNr(), sites, states, size);
    }

    /**
     * Recomputes the transition counts of the branch above the given node from scratch and
     * marks it as changed. This is meant for a change of the parent of the node, e.g. by a
     * topology move, which the incremental updates do not track.
     */
    public void markBranchChanged(Node node, Operator operator) {
        this.startEditing(operator);
        if (!node.isRoot()) {
            this.transitionCounts[node.getNr()] = this.computeTransitionCounts(node);
        }
        this.markBranchDirty(node.getNr());
    }

    private void markBranchDirty(int nodeNr) {
        if (!this.isBranchDirty[nodeNr]) {
            this.isBranchDirty[nodeNr] = true;
            this.dirtyNodeNrs[this.numDirtyNodes++] = nodeNr;
        }
    }

    /**
     * Replaces the root sequence and the changes of every node without going through an
     * operator. This is meant for initialisation only, so the new state is also stored. The
     * changes of the root must be empty.
     */
    public void initialiseStates(int[] rootSequence, List<StateChanges> changesAtNode) {
        if (this.hasStoredStates) {
            throw new IllegalStateException("Cannot initialise the states during a proposal.");
        }
        if (rootSequence.length != this.numSites) {
            throw new IllegalArgumentException("Expected a root sequence with " + this.numSites + " sites.");
        }
        if (changesAtNode.size() != this.numNodes) {
            throw new IllegalArgumentException("Expected changes for " + this.numNodes + " nodes.");
        }

        this.referenceSequence = rootSequence.clone();
        for (int nodeNr = 0; nodeNr < this.numNodes; nodeNr++) {
            this.changesAtNode[nodeNr] = changesAtNode.get(nodeNr);
            this.checkChanges(nodeNr);
        }
        this.recomputeAllCounts();
        this.markEveryBranchDirty();

        this.storeStates();
    }

    /* Copy on Write */

    /** Returns the transition counts above the given node, copying them first if they are shared with the stored state. */
    private int[] getEditableCounts(int nodeNr) {
        if (this.transitionCounts[nodeNr] == this.storedTransitionCounts[nodeNr]) {
            this.transitionCounts[nodeNr] = this.transitionCounts[nodeNr].clone();
        }
        return this.transitionCounts[nodeNr];
    }

    private int[] getEditableRootStateCounts() {
        if (this.rootStateCounts == this.storedRootStateCounts) {
            this.rootStateCounts = this.rootStateCounts.clone();
        }
        return this.rootStateCounts;
    }

    private int[] getEditableReferenceSequence() {
        if (this.referenceSequence == this.storedReferenceSequence) {
            this.referenceSequence = this.referenceSequence.clone();
        }
        return this.referenceSequence;
    }

    /* Sanity Checks */

    /**
     * Checks the changes and the transition counts of the given node and of its children
     * against a computation from scratch, as well as the root state counts.
     */
    private void performSanityChecks(Node node) {
        if (!Arrays.equals(this.rootStateCounts, this.computeRootStateCounts())) {
            throw new RuntimeException("Root state counts are inconsistent with the root sequence.");
        }

        this.checkChanges(node.getNr());
        this.checkTransitionCounts(node);
        for (Node child : node.getChildrenMutable()) {
            this.checkChanges(child.getNr());
            this.checkTransitionCounts(child);
        }
    }

    /**
     * Checks that the changes of the given node belong to it, have valid sites and states, and
     * only list sites at which the node differs from the root.
     */
    private void checkChanges(int nodeNr) {
        StateChanges changes = this.changesAtNode[nodeNr];

        if (changes.getNodeNr() != nodeNr) {
            throw new RuntimeException("Changes are annotated with a different node.");
        }
        if (this.tree.getNode(nodeNr).isRoot() && !changes.isEmpty()) {
            throw new RuntimeException("The root differs from the root sequence.");
        }

        for (int i = 0; i < changes.size(); i++) {
            int site = changes.getSite(i);
            int state = changes.getState(i);

            if (site >= this.numSites) {
                throw new RuntimeException("Change at site " + site + " is out of range.");
            }
            if (state < 0 || state >= this.numStates) {
                throw new RuntimeException("Change to state " + state + " is out of range.");
            }
            if (state == this.referenceSequence[site]) {
                throw new RuntimeException("Change at site " + site + " does not differ from the root sequence.");
            }
        }
    }

    private void checkTransitionCounts(Node node) {
        if (!node.isRoot() && !Arrays.equals(this.transitionCounts[node.getNr()], this.computeTransitionCounts(node))) {
            throw new RuntimeException("Transition counts above node " + node.getNr() + " are inconsistent with the states.");
        }
    }

    /* Getter */

    public StateChanges getChanges(Node node) {
        return this.changesAtNode[node.getNr()];
    }

    /** Returns the state of the given node at the given site, in O(log m) for m changes at the node. */
    public int getState(Node node, int site) {
        return this.changesAtNode[node.getNr()].getStateOrDefault(site, this.referenceSequence[site]);
    }

    /**
     * Returns the root sequence, to which the changes of every node refer. The array must not
     * be modified or kept, as it is replaced whenever the root sequence changes.
     */
    public int[] getReferenceSequence() {
        return this.referenceSequence;
    }

    /**
     * Returns the transition counts of the branch above the given non-root node, where entry
     * from * numStates + to counts the sites in state from at the parent and in state to at
     * the node. The array must not be modified or kept, as it is replaced when it changes.
     */
    public int[] getTransitionCounts(Node node) {
        if (node.isRoot()) {
            throw new IllegalArgumentException("The root has no branch with transition counts.");
        }
        return this.transitionCounts[node.getNr()];
    }

    /** Returns the number of sites of the root sequence in each state. The array must not be modified or kept. */
    public int[] getRootStateCounts() {
        return this.rootStateCounts;
    }

    public int getNumStates() {
        return this.numStates;
    }

    public int getNumSites() {
        return this.numSites;
    }

    /**
     * Returns whether the states at the given node or the transition counts above it may have
     * changed since the states were last marked clean, i.e. since the end of the previous MCMC
     * step.
     */
    public boolean isDirty(Node node) {
        return this.isBranchDirty[node.getNr()];
    }

    /**
     * Returns whether every branch may be dirty, in which case the dirty node numbers are not
     * tracked individually.
     */
    public boolean isEveryBranchDirty() {
        return this.isEveryBranchDirty;
    }

    /**
     * Returns the numbers of the nodes whose branch is dirty in the first getDirtyNodeCount()
     * entries. They are only complete if not every branch is dirty.
     */
    public int[] getDirtyNodeNrs() {
        return this.dirtyNodeNrs;
    }

    public int getDirtyNodeCount() {
        return this.numDirtyNodes;
    }

    /**
     * Returns the number of changes summed over all nodes, i.e. the number of sites at which a
     * node differs from the root sequence. This is the size of the stored state.
     */
    public int getChangeCount() {
        int count = 0;
        for (StateChanges changes : this.changesAtNode) {
            count += changes.size();
        }
        return count;
    }

    /**
     * Returns the number of differences summed over all branches, i.e. the number of sites at
     * which a node differs from its parent. This is a lower bound on the number of mutations.
     */
    public int getDifferenceCount() {
        int count = 0;
        for (Node node : this.tree.getNodesAsArray()) {
            if (node.isRoot()) {
                continue;
            }

            int[] counts = this.transitionCounts[node.getNr()];
            for (int from = 0; from < this.numStates; from++) {
                for (int to = 0; to < this.numStates; to++) {
                    if (from != to) {
                        count += counts[from * this.numStates + to];
                    }
                }
            }
        }
        return count;
    }

    public TreeInterface getTree() {
        return this.tree;
    }

    public Alignment getAlignment() {
        return this.alignment;
    }

    /* StateNode methods */

    /**
     * Marks the states as clean or dirty. MCMC calls this with false after every step,
     * whether it was accepted or rejected, which ends the current proposal.
     */
    @Override
    public void setEverythingDirty(boolean isDirty) {
        this.setSomethingIsDirty(isDirty);
        if (isDirty) {
            this.markEveryBranchDirty();
        } else {
            this.cleanBranches();
            this.hasStoredStates = false;
        }
    }

    private void markEveryBranchDirty() {
        Arrays.fill(this.isBranchDirty, true);
        this.isEveryBranchDirty = true;
        this.numDirtyNodes = 0;
    }

    private void cleanBranches() {
        if (this.isEveryBranchDirty) {
            Arrays.fill(this.isBranchDirty, false);
        } else {
            for (int i = 0; i < this.numDirtyNodes; i++) {
                this.isBranchDirty[this.dirtyNodeNrs[i]] = false;
            }
        }
        this.isEveryBranchDirty = false;
        this.numDirtyNodes = 0;
    }

    /**
     * Stores the states before a proposal changes them. The states take the tree as an input,
     * so BEAST also treats them as a calculation node downstream of the tree: after a proposal
     * that changes the tree, State calls store() and restore() on them a second time. Only the
     * first store() of a proposal takes a snapshot, as a later one would overwrite it with the
     * proposed states.
     */
    @Override
    protected void store() {
        if (!this.hasStoredStates) {
            this.storeStates();
            this.hasStoredStates = true;
        }
    }

    /**
     * Restores the states from before the proposal. Only the first restore() of a proposal
     * swaps back, as a second one would reinstate the rejected states.
     */
    @Override
    public void restore() {
        if (!this.hasStoredStates) {
            return;
        }

        int[] referenceSequence = this.referenceSequence;
        this.referenceSequence = this.storedReferenceSequence;
        this.storedReferenceSequence = referenceSequence;

        StateChanges[] changesAtNode = this.changesAtNode;
        this.changesAtNode = this.storedChangesAtNode;
        this.storedChangesAtNode = changesAtNode;

        int[][] transitionCounts = this.transitionCounts;
        this.transitionCounts = this.storedTransitionCounts;
        this.storedTransitionCounts = transitionCounts;

        int[] rootStateCounts = this.rootStateCounts;
        this.rootStateCounts = this.storedRootStateCounts;
        this.storedRootStateCounts = rootStateCounts;

        this.hasStoredStates = false;
    }

    /** Shares every array of the current state with the stored state, so that they are copied on their next write. */
    private void storeStates() {
        this.storedReferenceSequence = this.referenceSequence;
        System.arraycopy(this.changesAtNode, 0, this.storedChangesAtNode, 0, this.numNodes);
        System.arraycopy(this.transitionCounts, 0, this.storedTransitionCounts, 0, this.numNodes);
        this.storedRootStateCounts = this.rootStateCounts;
    }

    /** Returns a deep copy of the states. The counts are changed in place, so they are copied as well. */
    @Override
    public NodeStates copy() {
        try {
            NodeStates copy = (NodeStates) this.clone();
            copy.referenceSequence = this.referenceSequence.clone();
            copy.storedReferenceSequence = this.storedReferenceSequence.clone();
            copy.changesAtNode = this.changesAtNode.clone();
            copy.storedChangesAtNode = this.storedChangesAtNode.clone();
            copy.transitionCounts = deepCopy(this.transitionCounts);
            copy.storedTransitionCounts = deepCopy(this.storedTransitionCounts);
            copy.rootStateCounts = this.rootStateCounts.clone();
            copy.storedRootStateCounts = this.storedRootStateCounts.clone();
            copy.isBranchDirty = this.isBranchDirty.clone();
            copy.dirtyNodeNrs = this.dirtyNodeNrs.clone();
            return copy;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }

    private static int[][] deepCopy(int[][] arrays) {
        int[][] copy = new int[arrays.length][];
        for (int i = 0; i < arrays.length; i++) {
            copy[i] = arrays[i].clone();
        }
        return copy;
    }

    /** Assigns all values of this to the other states (other := this). */
    @Override
    public void assignTo(StateNode other) {
        NodeStates target = (NodeStates) other;
        target.setID(this.getID());
        target.index = this.index;
        target.copyValuesFrom(this);
    }

    /** Assigns all values of the other states to this (this := other). */
    @Override
    public void assignFrom(StateNode other) {
        NodeStates source = (NodeStates) other;
        this.setID(source.getID());
        this.copyValuesFrom(source);
    }

    /** Assigns only the root sequence and the changes of the other states to this, e.g. when resuming a run. */
    @Override
    public void assignFromFragile(StateNode other) {
        NodeStates source = (NodeStates) other;
        this.referenceSequence = source.referenceSequence.clone();
        System.arraycopy(source.changesAtNode, 0, this.changesAtNode, 0, this.numNodes);
        this.recomputeAllCounts();
        this.markEveryBranchDirty();
        this.setSomethingIsDirty(false);
    }

    /** Copies the tree, the alignment and the states of the given source into this. */
    private void copyValuesFrom(NodeStates source) {
        this.tree = source.tree;
        this.alignment = source.alignment;

        this.numStates = source.numStates;
        this.numNodes = source.numNodes;
        this.numSites = source.numSites;

        this.referenceSequence = source.referenceSequence.clone();
        this.storedReferenceSequence = source.storedReferenceSequence.clone();
        this.changesAtNode = source.changesAtNode.clone();
        this.storedChangesAtNode = source.storedChangesAtNode.clone();
        this.transitionCounts = deepCopy(source.transitionCounts);
        this.storedTransitionCounts = deepCopy(source.storedTransitionCounts);
        this.rootStateCounts = source.rootStateCounts.clone();
        this.storedRootStateCounts = source.storedRootStateCounts.clone();

        this.isBranchDirty = new boolean[this.numNodes];
        this.dirtyNodeNrs = new int[this.numNodes];
        this.markEveryBranchDirty();
    }

    /* Loggable methods */

    @Override
    public void init(PrintStream out) {
        out.print(this.getID() + ".count\t" + this.getID() + ".changes\t");
    }

    @Override
    public void log(long sample, PrintStream out) {
        out.print(this.getDifferenceCount() + "\t" + this.getChangeCount() + "\t");
    }

    @Override
    public void close(PrintStream out) {
        // nothing to close
    }

    /* Unsupported StateNode methods */

    @Override
    public void fromXML(org.w3c.dom.Node node) {
        throw new UnsupportedOperationException();
    }

}
