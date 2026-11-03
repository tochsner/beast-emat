package emat.initalisation;

/**
 * A rooted binary tree without branch lengths, given by the parent of every node. The tips are
 * the nodes 0 to numTips - 1, and the root is the only node without a parent.
 */
public class BinaryTopology {

    private final int numTips;
    private final int[] parents;
    private final int[][] children;
    private final int root;
    private final int[] preorder;

    public BinaryTopology(int numTips, int[] parents) {
        this.numTips = numTips;
        this.parents = parents.clone();
        this.children = this.collectChildren();
        this.root = this.findRoot();
        this.preorder = this.computePreorder();
    }

    /** Returns the children of every node, which are two for an internal node and none for a tip. */
    private int[][] collectChildren() {
        int[][] children = new int[this.parents.length][];
        int[] numChildren = new int[this.parents.length];
        for (int node = 0; node < this.parents.length; node++) {
            children[node] = new int[node < this.numTips ? 0 : 2];
        }

        for (int node = 0; node < this.parents.length; node++) {
            int parent = this.parents[node];
            if (parent >= 0) {
                children[parent][numChildren[parent]++] = node;
            }
        }
        return children;
    }

    private int findRoot() {
        for (int node = 0; node < this.parents.length; node++) {
            if (this.parents[node] < 0) {
                return node;
            }
        }
        throw new IllegalArgumentException("The topology has no root.");
    }

    /** Returns the nodes in an order where every node comes before its children. */
    private int[] computePreorder() {
        int[] preorder = new int[this.parents.length];
        int[] stack = new int[this.parents.length];
        int stackSize = 0;
        int numVisited = 0;

        stack[stackSize++] = this.root;
        while (stackSize > 0) {
            int node = stack[--stackSize];
            preorder[numVisited++] = node;
            for (int child : this.children[node]) {
                stack[stackSize++] = child;
            }
        }
        return preorder;
    }

    /* Accessors */

    public int getNumTips() {
        return this.numTips;
    }

    public int getNumNodes() {
        return this.parents.length;
    }

    /** Returns the parent of the node, or -1 for the root. */
    public int getParent(int node) {
        return this.parents[node];
    }

    public int[] getChildren(int node) {
        return this.children[node];
    }

    public int getRoot() {
        return this.root;
    }

    /** Returns the nodes in an order where every node comes before its children. */
    public int[] getPreorder() {
        return this.preorder;
    }

    /** Returns the nodes in an order where every node comes after its children. */
    public int[] getPostorder() {
        int[] postorder = this.preorder.clone();
        for (int i = 0, j = postorder.length - 1; i < j; i++, j--) {
            int node = postorder[i];
            postorder[i] = postorder[j];
            postorder[j] = node;
        }
        return postorder;
    }

}
