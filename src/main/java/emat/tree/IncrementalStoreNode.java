package emat.tree;

import beast.base.core.Description;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.List;

@Description("Tree node that reports its changes to an IncrementalStoreTree, so that the tree only " +
        "stores the nodes that changed.")
public class IncrementalStoreNode extends Node {

    @Override
    public Node copy() {
        return copyOf(this);
    }

    /**
     * Creates a copy of the given node and its subtree, made of incremental store nodes. The
     * subtree is traversed without recursion, so that deep trees such as the initial
     * caterpillar of a large taxon set do not overflow the stack.
     */
    static IncrementalStoreNode copyOf(Node source) {
        IncrementalStoreNode root = copyValuesOf(source);

        Deque<Node> sources = new ArrayDeque<>();
        Deque<IncrementalStoreNode> copies = new ArrayDeque<>();
        sources.push(source);
        copies.push(root);

        while (!sources.isEmpty()) {
            Node node = sources.pop();
            IncrementalStoreNode copy = copies.pop();

            for (Node child : node.getChildren()) {
                IncrementalStoreNode childCopy = copyValuesOf(child);
                copy.addChild(childCopy);
                sources.push(child);
                copies.push(childCopy);
            }
        }
        return root;
    }

    /**
     * Creates an incremental store node with the height, number, metadata and ID of the
     * given node, but without parent or children.
     */
    private static IncrementalStoreNode copyValuesOf(Node source) {
        IncrementalStoreNode node = new IncrementalStoreNode();
        node.height = source.getHeight();
        node.labelNr = source.getNr();
        node.metaDataString = source.metaDataString;
        node.lengthMetaDataString = source.lengthMetaDataString;
        for (String name : source.getMetaDataNames()) {
            node.metaData.put(name, source.getMetaData(name));
        }
        for (String name : source.getLengthMetaDataNames()) {
            node.lengthMetaData.put(name, source.getLengthMetaData(name));
        }
        node.setID(source.getID());
        return node;
    }

    /**
     * Copies the height, parent and children of the given node into this node, resolving
     * the referenced nodes by their numbers in the given array.
     */
    void assignStructure(IncrementalStoreNode source, Node[] nodes) {
        this.height = source.height;
        this.parent = source.parent == null ? null : nodes[source.parent.getNr()];

        List<Node> sourceChildren = source.children;
        if (this.children.size() == sourceChildren.size()) {
            for (int i = 0; i < sourceChildren.size(); i++) {
                IncrementalStoreNode child = (IncrementalStoreNode) nodes[sourceChildren.get(i).getNr()];
                child.parent = this;
                this.children.set(i, child);
            }
        } else {
            this.children.clear();
            for (Node sourceChild : sourceChildren) {
                IncrementalStoreNode child = (IncrementalStoreNode) nodes[sourceChild.getNr()];
                child.parent = this;
                this.children.add(child);
            }
        }
    }

    void clearDirty() {
        this.isDirty = Tree.IS_CLEAN;
    }

    @Override
    protected void startEditing() {
        // the tree stores itself when editing starts, so the change is marked afterwards
        super.startEditing();
        this.markChanged();
    }

    @Override
    public void setNr(int labelIndex) {
        super.setNr(labelIndex);
        if (this.m_tree instanceof IncrementalStoreTree tree) {
            tree.requireFullStore();
        }
    }

    @Override
    public void setHeight(double height) {
        super.setHeight(height);
        for (Node child : this.children) {
            ((IncrementalStoreNode) child).markFlagged();
        }
    }

    @Override
    public void makeDirty(int dirty) {
        super.makeDirty(dirty);
        this.markFlagged();
    }

    @Override
    public void makeAllDirty(int dirty) {
        super.makeAllDirty(dirty);
        this.markFlagged();
    }

    @Override
    public void setParent(Node parent, boolean inOperator) {
        super.setParent(parent, inOperator);
        this.markChanged();
    }

    @Override
    public void removeAllChildren(boolean inOperator) {
        super.removeAllChildren(inOperator);
        this.markChanged();
    }

    @Override
    public void addChild(Node child) {
        super.addChild(child);
        this.markChanged();
    }

    @Override
    public void setChild(int childIndex, Node node) {
        super.setChild(childIndex, node);
        this.markChanged();
    }

    @Override
    public void setLeft(Node leftChild) {
        super.setLeft(leftChild);
        this.markChanged();
    }

    @Override
    public void setRight(Node rightChild) {
        super.setRight(rightChild);
        this.markChanged();
    }

    private void markChanged() {
        if (this.m_tree instanceof IncrementalStoreTree tree) {
            tree.markNodeChanged(this.labelNr);
        }
    }

    private void markFlagged() {
        if (this.m_tree instanceof IncrementalStoreTree tree) {
            tree.markNodeFlagged(this.labelNr);
        }
    }
}
