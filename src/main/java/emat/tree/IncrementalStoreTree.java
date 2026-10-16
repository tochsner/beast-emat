package emat.tree;

import beast.base.core.Description;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.StateNode;

import java.util.ArrayDeque;
import java.util.Deque;

@Description("Tree that only stores and cleans the nodes changed since the last store instead of all " +
        "nodes, which makes storing and restoring independent of the tree size for operators that change " +
        "few nodes. Changes are tracked through the setters of IncrementalStoreNode, so the tree must not " +
        "be modified through the list returned by getChildrenMutable().")
public class IncrementalStoreTree extends Tree {

    /**
     * Nodes whose height, parent or children changed since the last store. As restore swaps the
     * current and stored nodes, these are also the nodes that differ after a rejected proposal.
     */
    private int[] changedNodeNrs;
    private boolean[] isNodeChanged;
    private int numChangedNodes;

    /**
     * Nodes whose dirtiness flags might have been set since they were last cleaned.
     */
    private int[] flaggedNodeNrs;
    private boolean[] isNodeFlagged;
    private int numFlaggedNodes;

    private boolean requiresFullStore = true;
    private boolean requiresFullClean = true;

    @Override
    protected Node newNode() {
        return new IncrementalStoreNode();
    }

    @Override
    public void initArrays() {
        if (!this.isIncremental(this.root)) {
            this.root = IncrementalStoreNode.copyOf(this.root);
        }
        super.initArrays();

        this.allocateTracking();
        this.requiresFullStore = true;
        this.requiresFullClean = true;
    }

    @Override
    public void assignFromFragile(StateNode other) {
        super.assignFromFragile(other);
        this.requiresFullStore = true;
        this.requiresFullClean = true;
    }

    @Override
    public void removeNode(int i) {
        super.removeNode(i);
        this.requiresFullStore = true;
        this.requiresFullClean = true;
    }

    @Override
    public void addNode(Node newNode) {
        super.addNode(newNode);
        this.requiresFullStore = true;
        this.requiresFullClean = true;
    }

    @Override
    protected void store() {
        if (this.requiresFullStore || this.m_storedNodes.length != this.nodeCount) {
            super.store();
            if (this.isNodeChanged.length != this.nodeCount) {
                this.allocateTracking();
                this.requiresFullClean = true;
            }
            this.requiresFullStore = false;
        } else {
            for (int i = 0; i < this.numChangedNodes; i++) {
                int nodeNr = this.changedNodeNrs[i];
                IncrementalStoreNode sink = (IncrementalStoreNode) this.m_storedNodes[nodeNr];
                sink.assignStructure((IncrementalStoreNode) this.m_nodes[nodeNr], this.m_storedNodes);
            }
            this.storedRoot = this.m_storedNodes[this.root.getNr()];
        }

        for (int i = 0; i < this.numChangedNodes; i++) {
            this.isNodeChanged[this.changedNodeNrs[i]] = false;
        }
        this.numChangedNodes = 0;
    }

    @Override
    public void restore() {
        if (this.m_storedNodes.length != this.nodeCount) {
            super.restore();
            this.requiresFullStore = true;
            this.requiresFullClean = true;
            return;
        }

        Node[] nodes = this.m_storedNodes;
        this.m_storedNodes = this.m_nodes;
        this.m_nodes = nodes;
        this.root = this.m_nodes[this.storedRoot.getNr()];

        this.hasStartedEditing = false;
        this.cleanFlaggedNodes();
        this.postCache = null;
    }

    @Override
    public void setEverythingDirty(boolean isDirty) {
        if (isDirty) {
            super.setEverythingDirty(true);
            this.requiresFullClean = true;
        } else {
            this.setSomethingIsDirty(false);
            this.cleanFlaggedNodes();
        }
    }

    void markNodeChanged(int nodeNr) {
        if (this.isNodeChanged == null || nodeNr < 0 || nodeNr >= this.isNodeChanged.length) {
            this.requireFullStore();
            return;
        }

        if (!this.isNodeChanged[nodeNr]) {
            this.isNodeChanged[nodeNr] = true;
            this.changedNodeNrs[this.numChangedNodes++] = nodeNr;
        }
        this.markNodeFlagged(nodeNr);
    }

    void markNodeFlagged(int nodeNr) {
        if (this.isNodeFlagged == null || nodeNr < 0 || nodeNr >= this.isNodeFlagged.length) {
            this.requiresFullClean = true;
            return;
        }

        if (!this.isNodeFlagged[nodeNr]) {
            this.isNodeFlagged[nodeNr] = true;
            this.flaggedNodeNrs[this.numFlaggedNodes++] = nodeNr;
        }
    }

    /**
     * Makes the next store copy all nodes, which is needed when nodes are renumbered.
     */
    void requireFullStore() {
        this.requiresFullStore = true;
        this.requiresFullClean = true;
    }

    /**
     * Cleans the flagged nodes in both the current and the stored nodes, so that the nodes
     * swapped in by a restore are clean as well.
     */
    private void cleanFlaggedNodes() {
        if (this.requiresFullClean) {
            this.cleanNodes(this.m_nodes);
            this.cleanNodes(this.m_storedNodes);
            this.requiresFullClean = false;
        } else {
            for (int i = 0; i < this.numFlaggedNodes; i++) {
                int nodeNr = this.flaggedNodeNrs[i];
                ((IncrementalStoreNode) this.m_nodes[nodeNr]).clearDirty();
                ((IncrementalStoreNode) this.m_storedNodes[nodeNr]).clearDirty();
            }
        }

        for (int i = 0; i < this.numFlaggedNodes; i++) {
            this.isNodeFlagged[this.flaggedNodeNrs[i]] = false;
        }
        this.numFlaggedNodes = 0;
    }

    private void cleanNodes(Node[] nodes) {
        for (Node node : nodes) {
            if (node != null) {
                ((IncrementalStoreNode) node).clearDirty();
            }
        }
    }

    private void allocateTracking() {
        this.changedNodeNrs = new int[this.nodeCount];
        this.isNodeChanged = new boolean[this.nodeCount];
        this.numChangedNodes = 0;

        this.flaggedNodeNrs = new int[this.nodeCount];
        this.isNodeFlagged = new boolean[this.nodeCount];
        this.numFlaggedNodes = 0;
    }

    /**
     * Checks whether all nodes of the subtree are incremental store nodes, without recursion
     * so that deep trees do not overflow the stack.
     */
    private boolean isIncremental(Node root) {
        Deque<Node> nodes = new ArrayDeque<>();
        nodes.push(root);
        while (!nodes.isEmpty()) {
            Node node = nodes.pop();
            if (!(node instanceof IncrementalStoreNode)) {
                return false;
            }
            node.getChildren().forEach(nodes::push);
        }
        return true;
    }
}
