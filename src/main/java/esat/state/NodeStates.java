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
 * Stores the explicit state of every node of a tree without materializing it. The root
 * sequence is stored in full, and every other node stores the sites at which it differs from
 * its parent. The state of a node is therefore the state of the nearest change at or above it,
 * and the state of the root sequence if there is none. The transition counts of every branch,
 * i.e. the number of sites in each pair of states at the parent and the node, are kept up to
 * date as the sufficient statistics of the branch.
 * <p>
 * Every edit keeps the states of all nodes it does not name, so an edit of a node only
 * rewrites the changes of the node and of its children. The changes are immutable and the
 * counts are copied on write: the stored state shares all arrays with the current state, and
 * an array is only copied before its first change in a proposal. Storing the state therefore
 * takes time linear in the number of nodes, independent of the number of states and changes.
 */
@Description("Stores the explicit states of every node of the given tree as changes compared to the parent node, " +
        "together with the transition counts of every branch.")
public class NodeStates extends StateNode {

    /** Receives the full sequence of every node during a traversal of the states. */
    public interface StateVisitor {

        /** Visits the given node with its full sequence, which must not be modified or kept. */
        void visit(Node node, int[] sequence);

    }

    final public Input<TreeInterface> treeInput = new Input<>("tree", "", Input.Validate.REQUIRED);
    final public Input<Alignment> alignmentInput = new Input<>("alignment", "", Input.Validate.REQUIRED);

    // the probability with which a change of the state is checked against a computation from scratch
    private static final double SANITY_CHECK_PROBABILITY = 0.01;

    TreeInterface tree;
    Alignment alignment;

    int numStates;
    int numNodes;
    int numSites;

    // the sequence of the root
    private int[] rootSequence;
    private int[] storedRootSequence;

    // per node, the sites at which the node differs from its parent, which is empty for the root
    private BranchChanges[] changesAtNode;
    private BranchChanges[] storedChangesAtNode;

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

    // whether the states of a tip may be incompatible with its data since the states were last marked clean, tracked like the dirty branches
    private boolean[] isCheckPending;
    private int[] checkNodeNrs;
    private int numCheckNodes;
    private boolean isEveryCheckPending;

    // whether the stored state holds a snapshot that the current proposal has not restored yet
    private boolean hasStoredStates = false;

    // the stamp of the sites replaced by the current merge, and the buffers of its result
    private int[] siteMarks;
    private int siteStamp;
    private int[] mergeSites = new int[0];
    private int[] mergeFromStates = new int[0];
    private int[] mergeToStates = new int[0];

    // the buffers of the composition of the changes along a path, indexed by site, and the sites touched by it
    private int[] pathStamps;
    private int pathStamp;
    private int[] pathFirstStates;
    private int[] pathLastStates;
    private int[] pathSites;
    private Node[] upwardNodes;
    private Node[] downwardNodes;

    // a copy of the states at the touched sites of a first path, for counts over two paths that start at the same node
    private int[] sidePathStamps;
    private int sidePathStamp;
    private int[] sidePathFirstStates;
    private int[] sidePathLastStates;
    private int[] sidePathSites;

    // scratch space for the state counts of a node and for the flags of the nodes that a replacement changes
    private int[] stateCounts;
    private boolean[] haveChangesChanged;
    private boolean[] haveStateCountsChanged;
    private Node[] preorder;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();
        this.alignment = this.alignmentInput.get();

        this.numStates = this.alignment.getMaxStateCount();
        this.numNodes = this.tree.getNodeCount();
        this.numSites = this.alignment.getSiteCount();

        this.allocateScratchSpace();

        // start with every node in the state of the first tip

        this.rootSequence = this.computeFirstTipSequence();

        this.changesAtNode = new BranchChanges[this.numNodes];
        for (int nodeNr = 0; nodeNr < this.numNodes; nodeNr++) {
            this.changesAtNode[nodeNr] = BranchChanges.empty(nodeNr);
        }

        this.transitionCounts = new int[this.numNodes][];
        this.recomputeAllCounts();

        this.isBranchDirty = new boolean[this.numNodes];
        this.dirtyNodeNrs = new int[this.numNodes];
        this.isCheckPending = new boolean[this.numNodes];
        this.checkNodeNrs = new int[this.numNodes];
        this.markEveryBranchDirty();
        this.markEveryCheckPending();

        this.storedChangesAtNode = new BranchChanges[this.numNodes];
        this.storedTransitionCounts = new int[this.numNodes][];
        this.storeStates();
    }

    private void allocateScratchSpace() {
        this.upwardNodes = new Node[this.numNodes];
        this.downwardNodes = new Node[this.numNodes];
        this.preorder = new Node[this.numNodes];
        this.stateCounts = new int[this.numStates];
        this.haveChangesChanged = new boolean[this.numNodes];
        this.haveStateCountsChanged = new boolean[this.numNodes];
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

        // the counts of a branch use the state counts of its parent, so parents come first
        int numPreorderNodes = this.collectNodesInPreorder();
        for (int i = 0; i < numPreorderNodes; i++) {
            Node node = this.preorder[i];
            this.transitionCounts[node.getNr()] = node.isRoot()
                    ? new int[this.numStates * this.numStates]
                    : this.computeBranchCounts(node);
        }
    }

    private int[] computeRootStateCounts() {
        int[] counts = new int[this.numStates];
        for (int state : this.rootSequence) {
            counts[state]++;
        }
        return counts;
    }

    /** Collects the nodes in preorder without recursion, which could overflow the stack on deep trees, and returns their number. */
    private int collectNodesInPreorder() {
        Node[] stack = new Node[this.numNodes];
        int stackSize = 0;
        int numCollected = 0;

        stack[stackSize++] = this.tree.getRoot();
        while (stackSize > 0) {
            Node node = stack[--stackSize];
            this.preorder[numCollected++] = node;
            for (int childIndex = 0; childIndex < node.getChildCount(); childIndex++) {
                Node child = node.getChild(childIndex);
                stack[stackSize++] = child;
            }
        }
        return numCollected;
    }

    /**
     * Computes the transition counts of the branch above the given non-root node from its
     * changes and from the state counts of its parent. The changes give the pairs of different
     * states, and the other sites of the parent are in the same state at the node.
     */
    private int[] computeBranchCounts(Node node) {
        int[] counts = new int[this.numStates * this.numStates];
        BranchChanges changes = this.changesAtNode[node.getNr()];
        for (int i = 0; i < changes.size(); i++) {
            counts[changes.getFrom(i) * this.numStates + changes.getTo(i)]++;
        }

        this.computeStateCounts(node.getParent(), this.stateCounts);
        this.fillDiagonal(counts, this.stateCounts);
        return counts;
    }

    /** Sets the diagonal of the given counts, such that each row sums to the given number of sites in its state at the upper node. */
    private void fillDiagonal(int[] counts, int[] upperStateCounts) {
        for (int state = 0; state < this.numStates; state++) {
            int numLeavingSites = 0;
            for (int other = 0; other < this.numStates; other++) {
                if (other != state) {
                    numLeavingSites += counts[state * this.numStates + other];
                }
            }
            counts[state * this.numStates + state] = upperStateCounts[state] - numLeavingSites;
        }
    }

    /** Writes the number of sites of the given node in each state into the given array, in O(K^2). */
    private void computeStateCounts(Node node, int[] result) {
        if (node.isRoot()) {
            System.arraycopy(this.rootStateCounts, 0, result, 0, this.numStates);
            return;
        }

        Arrays.fill(result, 0);
        int[] counts = this.transitionCounts[node.getNr()];
        for (int from = 0; from < this.numStates; from++) {
            for (int to = 0; to < this.numStates; to++) {
                result[to] += counts[from * this.numStates + to];
            }
        }
    }

    /* Paths */

    /**
     * Writes the transition counts between the states of any two nodes into the given array,
     * row-major by the state of the upper node, as if a branch ran between them. This composes
     * the changes along the path between them, so it takes O(the sizes of the changes on the path).
     */
    public void countTransitions(Node upper, Node lower, int[] counts) {
        int numTouchedSites = this.composePath(upper, lower);

        Arrays.fill(counts, 0);
        this.computeStateCounts(upper, this.stateCounts);
        for (int state = 0; state < this.numStates; state++) {
            counts[state * this.numStates + state] = this.stateCounts[state];
        }

        for (int i = 0; i < numTouchedSites; i++) {
            int site = this.pathSites[i];
            int from = this.pathFirstStates[site];
            int to = this.pathLastStates[site];
            if (from != to) {
                counts[from * this.numStates + from]--;
                counts[from * this.numStates + to]++;
            }
        }
    }

    /**
     * Writes the number of sites in each triple of states (d, c, x) of the given top, bottom
     * and side node into the given array of size K^3, indexed by (d K + c) K + x. The nodes can
     * be any three nodes of the tree, e.g. the ends of a branch and a node to attach to it.
     * This composes the changes along the paths from the top to the side and to the bottom
     * node, so it takes O(the sizes of the changes on both paths).
     */
    public void countTriples(Node top, Node bottom, Node side, int[] counts) {
        if (this.sidePathStamps == null) {
            this.sidePathStamps = new int[this.numSites];
            this.sidePathFirstStates = new int[this.numSites];
            this.sidePathLastStates = new int[this.numSites];
            this.sidePathSites = new int[this.numSites];
        }

        // keep the states of the side node at the sites touched by its path, as the second path reuses the buffers

        int numSideSites = this.composePath(top, side);
        this.sidePathStamp++;
        for (int i = 0; i < numSideSites; i++) {
            int site = this.pathSites[i];
            this.sidePathStamps[site] = this.sidePathStamp;
            this.sidePathFirstStates[site] = this.pathFirstStates[site];
            this.sidePathLastStates[site] = this.pathLastStates[site];
            this.sidePathSites[i] = site;
        }

        // every site starts in the triple of its state at the top

        Arrays.fill(counts, 0);
        this.computeStateCounts(top, this.stateCounts);
        for (int state = 0; state < this.numStates; state++) {
            counts[this.getTripleIndex(state, state, state)] = this.stateCounts[state];
        }

        int numBottomSites = this.composePath(top, bottom);
        for (int i = 0; i < numBottomSites; i++) {
            int site = this.pathSites[i];
            int topState = this.pathFirstStates[site];
            int sideState = this.sidePathStamps[site] == this.sidePathStamp ? this.sidePathLastStates[site] : topState;
            counts[this.getTripleIndex(topState, topState, topState)]--;
            counts[this.getTripleIndex(topState, this.pathLastStates[site], sideState)]++;
        }

        // the sites touched only by the path to the side keep the top state at the bottom
        for (int i = 0; i < numSideSites; i++) {
            int site = this.sidePathSites[i];
            if (this.pathStamps[site] != this.pathStamp) {
                int topState = this.sidePathFirstStates[site];
                counts[this.getTripleIndex(topState, topState, topState)]--;
                counts[this.getTripleIndex(topState, topState, this.sidePathLastStates[site])]++;
            }
        }
    }

    private int getTripleIndex(int topState, int bottomState, int sideState) {
        return (topState * this.numStates + bottomState) * this.numStates + sideState;
    }

    /**
     * Returns the changes of the lower node compared to the upper node, which can be any two
     * nodes of the tree, annotated with the number of the lower node. This is what the changes
     * of the lower node would be if it were a child of the upper node.
     */
    public BranchChanges composeChanges(Node upper, Node lower) {
        int numTouchedSites = this.composePath(upper, lower);
        Arrays.sort(this.pathSites, 0, numTouchedSites);

        int[] sites = new int[numTouchedSites];
        int[] fromStates = new int[numTouchedSites];
        int[] toStates = new int[numTouchedSites];
        int size = 0;
        for (int i = 0; i < numTouchedSites; i++) {
            int site = this.pathSites[i];
            if (this.pathFirstStates[site] != this.pathLastStates[site]) {
                sites[size] = site;
                fromStates[size] = this.pathFirstStates[site];
                toStates[size] = this.pathLastStates[site];
                size++;
            }
        }
        return BranchChanges.ofSorted(lower.getNr(), sites, fromStates, toStates, size);
    }

    /**
     * Composes the changes of the branches on the path from the upper to the lower node and
     * returns the number of sites that the path touches, which are listed in pathSites. For
     * every one, pathFirstStates gives the state of the upper and pathLastStates the state of
     * the lower node. Going up a branch swaps its states.
     */
    private int composePath(Node upper, Node lower) {
        if (this.pathStamps == null) {
            this.pathStamps = new int[this.numSites];
            this.pathFirstStates = new int[this.numSites];
            this.pathLastStates = new int[this.numSites];
            this.pathSites = new int[this.numSites];
        }
        this.pathStamp++;

        // climb from the lower of the two nodes until they meet, which visits only the nodes on the path

        int numUpward = 0;
        int numDownward = 0;
        Node up = upper;
        Node down = lower;
        while (up != down) {
            boolean isUpLower = up.getHeight() < down.getHeight() || (up.getHeight() == down.getHeight() && !up.isRoot());
            if (isUpLower) {
                this.upwardNodes[numUpward++] = up;
                up = up.getParent();
            } else {
                this.downwardNodes[numDownward++] = down;
                down = down.getParent();
            }
        }

        int numTouchedSites = 0;
        for (int k = 0; k < numUpward; k++) {
            BranchChanges changes = this.changesAtNode[this.upwardNodes[k].getNr()];
            for (int i = 0; i < changes.size(); i++) {
                numTouchedSites = this.appendPathStep(numTouchedSites, changes.getSite(i), changes.getTo(i), changes.getFrom(i));
            }
        }
        for (int k = numDownward - 1; k >= 0; k--) {
            BranchChanges changes = this.changesAtNode[this.downwardNodes[k].getNr()];
            for (int i = 0; i < changes.size(); i++) {
                numTouchedSites = this.appendPathStep(numTouchedSites, changes.getSite(i), changes.getFrom(i), changes.getTo(i));
            }
        }
        return numTouchedSites;
    }

    private int appendPathStep(int numTouchedSites, int site, int startState, int endState) {
        if (this.pathStamps[site] != this.pathStamp) {
            this.pathStamps[site] = this.pathStamp;
            this.pathFirstStates[site] = startState;
            this.pathSites[numTouchedSites++] = site;
        }
        this.pathLastStates[site] = endState;
        return numTouchedSites;
    }

    /* State Management */

    /**
     * Sets the states of the given node at the given sites, which must be strictly ascending,
     * and keeps the states of every other node. The changes of the node and of its children are
     * rewritten at these sites, and the transition counts above them are updated, so this
     * costs O(n (c log m + depth log m)) for n sites and c children. The states of an inner
     * node are not checked, and a tip is marked for a check against its data.
     */
    public void setStates(Node node, int[] sites, int[] newStates, Operator operator) {
        if (sites.length != newStates.length) {
            throw new IllegalArgumentException("Expected as many states as sites.");
        }
        this.checkAscending(sites, sites.length);

        this.startEditing(operator);

        // per edited node, the new entries at the edited sites, where entries that equal the state above are left out
        int numChildren = node.getChildCount();
        int[][] editedSites = new int[numChildren + 1][sites.length];
        int[][] newFromStates = new int[numChildren + 1][sites.length];
        int[][] newToStates = new int[numChildren + 1][sites.length];
        int[] numEdited = new int[numChildren + 1];

        for (int i = 0; i < sites.length; i++) {
            int site = sites[i];
            int newState = newStates[i];
            if (newState < 0 || newState >= this.numStates) {
                throw new IllegalArgumentException("State " + newState + " is out of range.");
            }

            int oldState = this.getState(node, site);
            if (oldState == newState) {
                continue;
            }

            this.moveSiteState(node, site, oldState, newState, editedSites, newFromStates, newToStates, numEdited);
        }

        this.installEditedChanges(node, editedSites, newFromStates, newToStates, numEdited);

        this.markBranchDirty(node.getNr());
        for (Node child : node.getChildrenMutable()) {
            this.markBranchDirty(child.getNr());
        }
        if (node.isLeaf()) {
            this.markCheckPending(node.getNr());
        }

        // perform sanity checks from time to time
        if (Randomizer.nextDouble() < SANITY_CHECK_PROBABILITY) {
            this.performSanityChecks(node);
        }
    }

    /**
     * Updates the transition counts of the branch above the given node and of the branches
     * above its children for a change of the state of the node at the given site, and records
     * the new entries of these branches at the site. The index 0 of the recorded entries is the
     * node and the indices after it are its children.
     */
    private void moveSiteState(Node node, int site, int oldState, int newState, int[][] editedSites, int[][] newFromStates, int[][] newToStates, int[] numEdited) {
        if (node.isRoot()) {
            int[] rootStateCounts = this.getEditableRootStateCounts();
            rootStateCounts[oldState]--;
            rootStateCounts[newState]++;
            this.getEditableRootSequence()[site] = newState;
        } else {
            // without an entry, the node equals its parent
            BranchChanges changes = this.changesAtNode[node.getNr()];
            int index = changes.indexOfSite(site);
            int parentState = index >= 0 ? changes.getFrom(index) : oldState;

            int[] counts = this.getEditableCounts(node.getNr());
            counts[parentState * this.numStates + oldState]--;
            counts[parentState * this.numStates + newState]++;
            this.recordEntry(0, site, parentState, newState, editedSites, newFromStates, newToStates, numEdited);
        }

        int childIndex = 1;
        for (Node child : node.getChildrenMutable()) {
            BranchChanges childChanges = this.changesAtNode[child.getNr()];
            int index = childChanges.indexOfSite(site);
            int childState = index >= 0 ? childChanges.getTo(index) : oldState;

            int[] childCounts = this.getEditableCounts(child.getNr());
            childCounts[oldState * this.numStates + childState]--;
            childCounts[newState * this.numStates + childState]++;
            this.recordEntry(childIndex, site, newState, childState, editedSites, newFromStates, newToStates, numEdited);
            childIndex++;
        }
    }

    private void recordEntry(int nodeIndex, int site, int fromState, int toState, int[][] editedSites, int[][] newFromStates, int[][] newToStates, int[] numEdited) {
        // an entry that equals the state above is no change, but its site is still replaced
        editedSites[nodeIndex][numEdited[nodeIndex]] = site;
        newFromStates[nodeIndex][numEdited[nodeIndex]] = fromState;
        newToStates[nodeIndex][numEdited[nodeIndex]] = toState;
        numEdited[nodeIndex]++;
    }

    /** Replaces the entries of the given node and of its children at their edited sites. */
    private void installEditedChanges(Node node, int[][] editedSites, int[][] newFromStates, int[][] newToStates, int[] numEdited) {
        if (!node.isRoot()) {
            this.installEditedChanges(node, editedSites[0], newFromStates[0], newToStates[0], numEdited[0]);
        }

        int childIndex = 1;
        for (Node child : node.getChildrenMutable()) {
            this.installEditedChanges(child, editedSites[childIndex], newFromStates[childIndex], newToStates[childIndex], numEdited[childIndex]);
            childIndex++;
        }
    }

    private void installEditedChanges(Node node, int[] sites, int[] fromStates, int[] toStates, int numEdited) {
        if (numEdited == 0) {
            return;
        }

        this.markSites(sites, numEdited);

        int[] newSites = new int[numEdited];
        int[] newFromStates = new int[numEdited];
        int[] newToStates = new int[numEdited];
        int size = 0;
        for (int i = 0; i < numEdited; i++) {
            if (fromStates[i] != toStates[i]) {
                newSites[size] = sites[i];
                newFromStates[size] = fromStates[i];
                newToStates[size] = toStates[i];
                size++;
            }
        }

        BranchChanges newChanges = BranchChanges.ofSorted(node.getNr(), newSites, newFromStates, newToStates, size);
        this.changesAtNode[node.getNr()] = this.mergeChangesAtMarkedSites(this.changesAtNode[node.getNr()], newChanges);
    }

    /**
     * Installs the given changes for the given non-root node and recomputes the transition
     * counts above it from them. This is meant for a change of the parent of the node, e.g. by
     * a topology move, where the changes were composed before the move, so the states of the
     * node stay the same and a tip is not rechecked against its data. Parents must be
     * installed before their children if both change, as the counts use the state counts of the parent.
     */
    public void setBranchChanges(Node node, BranchChanges changes, Operator operator) {
        if (node.isRoot()) {
            throw new IllegalArgumentException("The root has no changes.");
        }
        if (changes.getNodeNr() != node.getNr()) {
            throw new IllegalArgumentException("Changes are annotated with a different node.");
        }

        this.startEditing(operator);
        this.changesAtNode[node.getNr()] = changes;
        this.transitionCounts[node.getNr()] = this.computeBranchCounts(node);
        this.markBranchDirty(node.getNr());
    }

    private void markBranchDirty(int nodeNr) {
        if (!this.isBranchDirty[nodeNr]) {
            this.isBranchDirty[nodeNr] = true;
            this.dirtyNodeNrs[this.numDirtyNodes++] = nodeNr;
        }
    }

    private void markCheckPending(int nodeNr) {
        if (!this.isCheckPending[nodeNr]) {
            this.isCheckPending[nodeNr] = true;
            this.checkNodeNrs[this.numCheckNodes++] = nodeNr;
        }
    }

    /**
     * Replaces the root sequence and the changes of every node without going through an
     * operator. This is meant for initialisation only, so the new state is also stored. The
     * changes of the root must be empty.
     */
    public void initialiseStates(int[] rootSequence, List<BranchChanges> changesAtNode) {
        if (this.hasStoredStates) {
            throw new IllegalStateException("Cannot initialise the states during a proposal.");
        }

        this.setAllStates(rootSequence, changesAtNode.toArray(new BranchChanges[0]));
        this.storeStates();
    }

    /**
     * Replaces the root sequence and the changes of every node within a proposal of the given
     * operator, and recomputes every transition count. The changes of the root must be empty.
     * The new counts are fresh arrays, so the stored state stays intact.
     */
    public void replaceStates(int[] rootSequence, BranchChanges[] changesAtNode, Operator operator) {
        this.startEditing(operator);
        this.setAllStates(rootSequence, changesAtNode);
    }

    private void setAllStates(int[] rootSequence, BranchChanges[] changesAtNode) {
        if (rootSequence.length != this.numSites) {
            throw new IllegalArgumentException("Expected a root sequence with " + this.numSites + " sites.");
        }
        if (changesAtNode.length != this.numNodes) {
            throw new IllegalArgumentException("Expected changes for " + this.numNodes + " nodes.");
        }

        for (int state : rootSequence) {
            if (state < 0 || state >= this.numStates) {
                throw new IllegalArgumentException("State " + state + " is out of range.");
            }
        }

        this.rootSequence = rootSequence.clone();
        for (int nodeNr = 0; nodeNr < this.numNodes; nodeNr++) {
            this.changesAtNode[nodeNr] = changesAtNode[nodeNr];
            this.checkChanges(nodeNr);
        }

        // the traversal throws if a change does not start in the state of its parent
        this.visitStates((node, sequence) -> {
        });

        this.recomputeAllCounts();
        this.markEveryBranchDirty();
        this.markEveryCheckPending();
    }

    /**
     * Replaces the states of every node at the given sites, which must be strictly ascending,
     * by the given root states and the changes of every node at these sites, which may only
     * list the given sites and are empty for the root. The changes have to be consistent, i.e.
     * start in the state of the parent at the site. The states at all other sites stay the
     * same. The counts are only recomputed next to the nodes whose changes differ, unless the
     * root states change, and only the nodes whose changes or counts above them change are
     * marked dirty. This takes O(n K^2 + m) for the m old and new changes of all nodes at
     * the sites, independently of the number of sites and of the sizes of the other changes.
     * The states of the tips are not checked against their data.
     */
    public void replaceStatesAtSites(int[] sites, int[] rootStates, BranchChanges[] changesAtSites, Operator operator) {
        if (sites.length != rootStates.length) {
            throw new IllegalArgumentException("Expected as many root states as sites.");
        }
        if (changesAtSites.length != this.numNodes) {
            throw new IllegalArgumentException("Expected changes for " + this.numNodes + " nodes.");
        }

        this.startEditing(operator);
        this.markSites(sites, sites.length);

        boolean hasRootChanged = this.replaceRootStates(sites, rootStates);

        // replace the changes of every node at the sites, keeping them if nothing differs

        Node root = this.tree.getRoot();
        Arrays.fill(this.haveChangesChanged, false);
        for (int nodeNr = 0; nodeNr < this.numNodes; nodeNr++) {
            if (nodeNr == root.getNr()) {
                if (!changesAtSites[nodeNr].isEmpty()) {
                    throw new IllegalArgumentException("The root has no changes, its states are given by the root states.");
                }
                continue;
            }

            BranchChanges oldChanges = this.changesAtNode[nodeNr];
            BranchChanges newChanges = this.mergeChangesAtMarkedSites(oldChanges, changesAtSites[nodeNr]);
            if (newChanges != oldChanges) {
                this.changesAtNode[nodeNr] = newChanges;
                this.haveChangesChanged[nodeNr] = true;
            }
        }

        this.recomputeCountsNextToChangedNodes(hasRootChanged);

        // perform sanity checks from time to time
        if (Randomizer.nextDouble() < SANITY_CHECK_PROBABILITY) {
            this.performSanityChecks(this.tree.getNode(Randomizer.nextInt(this.numNodes)));
        }
    }

    /** Sets the root states at the given sites, updates the root state counts, and returns whether any state changed. */
    private boolean replaceRootStates(int[] sites, int[] rootStates) {
        boolean hasRootChanged = false;
        for (int i = 0; i < sites.length; i++) {
            int oldState = this.rootSequence[sites[i]];
            if (rootStates[i] != oldState) {
                this.getEditableRootSequence()[sites[i]] = rootStates[i];
                int[] rootStateCounts = this.getEditableRootStateCounts();
                rootStateCounts[oldState]--;
                rootStateCounts[rootStates[i]]++;
                hasRootChanged = true;
            }
        }
        if (hasRootChanged) {
            this.markBranchDirty(this.tree.getRoot().getNr());
        }
        return hasRootChanged;
    }

    /**
     * Recomputes the counts of the branches whose changes changed or whose parent has other
     * state counts than before, from the parents down, and marks them dirty if they differ.
     */
    private void recomputeCountsNextToChangedNodes(boolean hasRootChanged) {
        Arrays.fill(this.haveStateCountsChanged, false);
        this.haveStateCountsChanged[this.tree.getRoot().getNr()] = hasRootChanged;

        int numPreorderNodes = this.collectNodesInPreorder();
        int[] oldStateCounts = new int[this.numStates];
        for (int i = 1; i < numPreorderNodes; i++) {
            Node node = this.preorder[i];
            int nodeNr = node.getNr();
            if (!this.haveChangesChanged[nodeNr] && !this.haveStateCountsChanged[node.getParent().getNr()]) {
                continue;
            }

            this.computeStateCounts(node, oldStateCounts);
            int[] newCounts = this.computeBranchCounts(node);
            boolean haveCountsChanged = !Arrays.equals(newCounts, this.transitionCounts[nodeNr]);
            if (haveCountsChanged) {
                this.transitionCounts[nodeNr] = newCounts;
                this.computeStateCounts(node, this.stateCounts);
                this.haveStateCountsChanged[nodeNr] = !Arrays.equals(oldStateCounts, this.stateCounts);
            }

            if (haveCountsChanged || this.haveChangesChanged[nodeNr]) {
                this.markBranchDirty(nodeNr);
            }
        }
    }

    /** Marks the given sites, which must be strictly ascending, for the membership tests of the merge. */
    private void markSites(int[] sites, int numSites) {
        if (this.siteMarks == null) {
            this.siteMarks = new int[this.numSites];
        }

        this.checkAscending(sites, numSites);

        // a new stamp unmarks all earlier sites at once
        this.siteStamp++;
        for (int i = 0; i < numSites; i++) {
            this.siteMarks[sites[i]] = this.siteStamp;
        }
    }

    private void checkAscending(int[] sites, int numSites) {
        for (int i = 0; i < numSites; i++) {
            if (sites[i] < 0 || sites[i] >= this.numSites) {
                throw new IllegalArgumentException("Site " + sites[i] + " is out of range.");
            }
            if (i > 0 && sites[i] <= sites[i - 1]) {
                throw new IllegalArgumentException("Sites must be strictly ascending.");
            }
        }
    }

    /**
     * Returns the given old changes with their entries at the marked sites replaced by the
     * given new changes, or the old changes themselves if the result is the same.
     */
    private BranchChanges mergeChangesAtMarkedSites(BranchChanges oldChanges, BranchChanges newChanges) {
        int maxSize = oldChanges.size() + newChanges.size();
        if (this.mergeSites.length < maxSize) {
            this.mergeSites = new int[2 * maxSize];
            this.mergeFromStates = new int[2 * maxSize];
            this.mergeToStates = new int[2 * maxSize];
        }

        int i = 0;
        int j = 0;
        int size = 0;
        while (i < oldChanges.size() || j < newChanges.size()) {
            int oldSite = i < oldChanges.size() ? oldChanges.getSite(i) : Integer.MAX_VALUE;
            int newSite = j < newChanges.size() ? newChanges.getSite(j) : Integer.MAX_VALUE;

            if (newSite <= oldSite) {
                if (this.siteMarks[newSite] != this.siteStamp) {
                    throw new IllegalArgumentException("Change at site " + newSite + " is not among the replaced sites.");
                }
                this.mergeSites[size] = newSite;
                this.mergeFromStates[size] = newChanges.getFrom(j);
                this.mergeToStates[size] = newChanges.getTo(j);
                size++;
                j++;
                if (newSite == oldSite) {
                    i++;
                }
            } else {
                // old entries at the replaced sites are dropped
                if (this.siteMarks[oldSite] != this.siteStamp) {
                    this.mergeSites[size] = oldSite;
                    this.mergeFromStates[size] = oldChanges.getFrom(i);
                    this.mergeToStates[size] = oldChanges.getTo(i);
                    size++;
                }
                i++;
            }
        }

        if (size == oldChanges.size() && this.areMergedEntriesEqual(oldChanges, size)) {
            return oldChanges;
        }
        return BranchChanges.ofSorted(oldChanges.getNodeNr(), this.mergeSites, this.mergeFromStates, this.mergeToStates, size);
    }

    private boolean areMergedEntriesEqual(BranchChanges changes, int size) {
        for (int k = 0; k < size; k++) {
            if (this.mergeSites[k] != changes.getSite(k)
                    || this.mergeFromStates[k] != changes.getFrom(k)
                    || this.mergeToStates[k] != changes.getTo(k)) {
                return false;
            }
        }
        return true;
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

    private int[] getEditableRootSequence() {
        if (this.rootSequence == this.storedRootSequence) {
            this.rootSequence = this.rootSequence.clone();
        }
        return this.rootSequence;
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

        this.checkBranch(node);
        for (Node child : node.getChildrenMutable()) {
            this.checkBranch(child);
        }
    }

    private void checkBranch(Node node) {
        this.checkChanges(node.getNr());
        if (node.isRoot()) {
            return;
        }

        // every change must start in the state of the parent
        BranchChanges changes = this.changesAtNode[node.getNr()];
        for (int i = 0; i < changes.size(); i++) {
            if (changes.getFrom(i) != this.getState(node.getParent(), changes.getSite(i))) {
                throw new RuntimeException("Change at site " + changes.getSite(i) + " of node " + node.getNr() + " does not start in the state of the parent.");
            }
        }

        if (!Arrays.equals(this.transitionCounts[node.getNr()], this.computeBranchCounts(node))) {
            throw new RuntimeException("Transition counts above node " + node.getNr() + " are inconsistent with the states.");
        }
    }

    /** Checks that the changes of the given node belong to it and have valid sites and states, which are empty for the root. */
    private void checkChanges(int nodeNr) {
        BranchChanges changes = this.changesAtNode[nodeNr];

        if (changes.getNodeNr() != nodeNr) {
            throw new RuntimeException("Changes are annotated with a different node.");
        }
        if (this.tree.getNode(nodeNr).isRoot() && !changes.isEmpty()) {
            throw new RuntimeException("The root has changes compared to a parent.");
        }

        for (int i = 0; i < changes.size(); i++) {
            int site = changes.getSite(i);

            if (site >= this.numSites) {
                throw new RuntimeException("Change at site " + site + " is out of range.");
            }
            for (int state : new int[]{changes.getFrom(i), changes.getTo(i)}) {
                if (state < 0 || state >= this.numStates) {
                    throw new RuntimeException("Change to state " + state + " is out of range.");
                }
            }
        }
    }

    /* Getter */

    public BranchChanges getChanges(Node node) {
        return this.changesAtNode[node.getNr()];
    }

    /**
     * Returns the state of the given node at the given site, which is the state of the nearest
     * change at or above the node. This takes O(depth log m), so it is meant for single lookups.
     */
    public int getState(Node node, int site) {
        for (Node current = node; !current.isRoot(); current = current.getParent()) {
            BranchChanges changes = this.changesAtNode[current.getNr()];
            int index = changes.indexOfSite(site);
            if (index >= 0) {
                return changes.getTo(index);
            }
        }
        return this.rootSequence[site];
    }

    /** Returns the full sequence of the given node as a new array, in O(L + depth m). */
    public int[] materializeState(Node node) {
        int numAncestors = 0;
        for (Node current = node; !current.isRoot(); current = current.getParent()) {
            this.upwardNodes[numAncestors++] = current;
        }

        int[] sequence = this.rootSequence.clone();
        for (int k = numAncestors - 1; k >= 0; k--) {
            BranchChanges changes = this.changesAtNode[this.upwardNodes[k].getNr()];
            for (int i = 0; i < changes.size(); i++) {
                sequence[changes.getSite(i)] = changes.getTo(i);
            }
        }
        return sequence;
    }

    /**
     * Calls the visitor with the full sequence of every node, parents before their children.
     * The sequence is shared and updated in place along the way, so this takes O(L + m) for
     * all m changes. It throws if a change does not start in the state of its parent.
     */
    public void visitStates(StateVisitor visitor) {
        int[] sequence = this.rootSequence.clone();
        Node root = this.tree.getRoot();

        // the nodes on the current path from the root, and the index of the next child to visit at each
        Node[] path = new Node[this.numNodes];
        int[] nextChildIndices = new int[this.numNodes];
        int depth = 0;
        path[0] = root;
        nextChildIndices[0] = 0;
        visitor.visit(root, sequence);

        while (depth >= 0) {
            Node node = path[depth];
            if (nextChildIndices[depth] < node.getChildCount()) {
                Node child = node.getChild(nextChildIndices[depth]++);
                this.applyChanges(this.changesAtNode[child.getNr()], sequence);
                depth++;
                path[depth] = child;
                nextChildIndices[depth] = 0;
                visitor.visit(child, sequence);
            } else {
                if (depth > 0) {
                    this.revertChanges(this.changesAtNode[node.getNr()], sequence);
                }
                depth--;
            }
        }
    }

    private void applyChanges(BranchChanges changes, int[] sequence) {
        for (int i = 0; i < changes.size(); i++) {
            if (sequence[changes.getSite(i)] != changes.getFrom(i)) {
                throw new IllegalStateException("Change at site " + changes.getSite(i) + " of node " + changes.getNodeNr() + " does not start in the state of the parent.");
            }
            sequence[changes.getSite(i)] = changes.getTo(i);
        }
    }

    private void revertChanges(BranchChanges changes, int[] sequence) {
        for (int i = 0; i < changes.size(); i++) {
            sequence[changes.getSite(i)] = changes.getFrom(i);
        }
    }

    /** Returns the sequence of the root, which must not be modified or kept, as it is replaced whenever it changes. */
    public int[] getRootSequence() {
        return this.rootSequence;
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

    /** Returns the number of sites of the given node in each state as a new array, in O(K^2). */
    public int[] getStateCounts(Node node) {
        int[] result = new int[this.numStates];
        this.computeStateCounts(node, result);
        return result;
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
     * Returns whether the states of every tip may be incompatible with its data, which is the
     * case if all states were replaced. Otherwise, only the tips listed by getCheckNodeNrs() may
     * be incompatible.
     */
    public boolean isEveryCheckPending() {
        return this.isEveryCheckPending;
    }

    /** Returns the numbers of the tips whose states were set directly since the states were last marked clean, in the first getCheckNodeCount() entries. */
    public int[] getCheckNodeNrs() {
        return this.checkNodeNrs;
    }

    public int getCheckNodeCount() {
        return this.numCheckNodes;
    }

    /**
     * Returns the number of differences summed over all branches, i.e. the number of sites at
     * which a node differs from its parent. This is a lower bound on the number of mutations
     * and the size of the stored state.
     */
    public int getDifferenceCount() {
        int count = 0;
        for (BranchChanges changes : this.changesAtNode) {
            count += changes.size();
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
            // the states are unchanged, so the tips stay compatible with their data
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

    /** Marks the states of every tip for a check against its data, as all states were replaced. */
    private void markEveryCheckPending() {
        Arrays.fill(this.isCheckPending, true);
        this.isEveryCheckPending = true;
        this.numCheckNodes = 0;
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

        if (this.isEveryCheckPending) {
            Arrays.fill(this.isCheckPending, false);
        } else {
            for (int i = 0; i < this.numCheckNodes; i++) {
                this.isCheckPending[this.checkNodeNrs[i]] = false;
            }
        }
        this.isEveryCheckPending = false;
        this.numCheckNodes = 0;
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

        int[] rootSequence = this.rootSequence;
        this.rootSequence = this.storedRootSequence;
        this.storedRootSequence = rootSequence;

        BranchChanges[] changesAtNode = this.changesAtNode;
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
        this.storedRootSequence = this.rootSequence;
        System.arraycopy(this.changesAtNode, 0, this.storedChangesAtNode, 0, this.numNodes);
        System.arraycopy(this.transitionCounts, 0, this.storedTransitionCounts, 0, this.numNodes);
        this.storedRootStateCounts = this.rootStateCounts;
    }

    /** Returns a deep copy of the states. The counts are changed in place, so they are copied as well. */
    @Override
    public NodeStates copy() {
        try {
            NodeStates copy = (NodeStates) this.clone();
            copy.rootSequence = this.rootSequence.clone();
            copy.storedRootSequence = this.storedRootSequence.clone();
            copy.changesAtNode = this.changesAtNode.clone();
            copy.storedChangesAtNode = this.storedChangesAtNode.clone();
            copy.transitionCounts = deepCopy(this.transitionCounts);
            copy.storedTransitionCounts = deepCopy(this.storedTransitionCounts);
            copy.rootStateCounts = this.rootStateCounts.clone();
            copy.storedRootStateCounts = this.storedRootStateCounts.clone();
            copy.isBranchDirty = this.isBranchDirty.clone();
            copy.dirtyNodeNrs = this.dirtyNodeNrs.clone();
            copy.isCheckPending = this.isCheckPending.clone();
            copy.checkNodeNrs = this.checkNodeNrs.clone();

            // the scratch space must not be shared with the original
            copy.siteMarks = null;
            copy.pathStamps = null;
            copy.pathFirstStates = null;
            copy.pathLastStates = null;
            copy.pathSites = null;
            copy.mergeSites = new int[0];
            copy.mergeFromStates = new int[0];
            copy.mergeToStates = new int[0];
            copy.allocateScratchSpace();
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
        this.rootSequence = source.rootSequence.clone();
        System.arraycopy(source.changesAtNode, 0, this.changesAtNode, 0, this.numNodes);
        this.recomputeAllCounts();
        this.markEveryBranchDirty();
        this.markEveryCheckPending();
        this.setSomethingIsDirty(false);
    }

    /** Copies the tree, the alignment and the states of the given source into this. */
    private void copyValuesFrom(NodeStates source) {
        this.tree = source.tree;
        this.alignment = source.alignment;

        this.numStates = source.numStates;
        this.numNodes = source.numNodes;
        this.numSites = source.numSites;
        this.allocateScratchSpace();
        this.siteMarks = null;
        this.pathStamps = null;

        this.rootSequence = source.rootSequence.clone();
        this.storedRootSequence = source.storedRootSequence.clone();
        this.changesAtNode = source.changesAtNode.clone();
        this.storedChangesAtNode = source.storedChangesAtNode.clone();
        this.transitionCounts = deepCopy(source.transitionCounts);
        this.storedTransitionCounts = deepCopy(source.storedTransitionCounts);
        this.rootStateCounts = source.rootStateCounts.clone();
        this.storedRootStateCounts = source.storedRootStateCounts.clone();

        this.isBranchDirty = new boolean[this.numNodes];
        this.dirtyNodeNrs = new int[this.numNodes];
        this.isCheckPending = new boolean[this.numNodes];
        this.checkNodeNrs = new int[this.numNodes];
        this.markEveryBranchDirty();
        this.markEveryCheckPending();
    }

    /* Loggable methods */

    @Override
    public void init(PrintStream out) {
        out.print(this.getID() + ".count\t");
    }

    @Override
    public void log(long sample, PrintStream out) {
        out.print(this.getDifferenceCount() + "\t");
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
