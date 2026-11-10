package esat.stochasticmapping;

import esat.state.BranchChanges;

/**
 * A cursor over the union of the sites of the changes of the branches around a node, in
 * ascending order: the branch above the node, unless it is the root, and the branches above
 * its children. At every site of the union, it gives the state of each neighbour, i.e. of the
 * parent and of the children, and the current state of the node. A neighbour without a change
 * at the site is in the state of the node. A step costs O(k) for k lists, and the arrays are
 * reused between merges.
 */
final class ChangeListMerge {

    private BranchChanges[] lists = new BranchChanges[0];
    private int numLists = 0;
    private boolean hasParent;

    // per list: the index of its next unvisited change, and the index of its change at the current site or -1
    private int[] nextIndices = new int[0];
    private int[] currentIndices = new int[0];

    private int currentState;

    /**
     * Starts a merge of the first given number of lists. If the node has a parent, the first
     * list is the changes of the node and the others are those of its children, otherwise all
     * lists are those of its children.
     */
    void reset(BranchChanges[] lists, int numLists, boolean hasParent) {
        if (this.lists.length < numLists) {
            this.lists = new BranchChanges[numLists];
            this.nextIndices = new int[numLists];
            this.currentIndices = new int[numLists];
        }

        System.arraycopy(lists, 0, this.lists, 0, numLists);
        this.numLists = numLists;
        this.hasParent = hasParent;
        for (int i = 0; i < numLists; i++) {
            this.nextIndices[i] = 0;
            this.currentIndices[i] = -1;
        }
    }

    /** Moves to the next site of the union and returns it, or returns Integer.MAX_VALUE if no site is left. */
    int nextSite() {
        int nextSite = Integer.MAX_VALUE;
        for (int i = 0; i < this.numLists; i++) {
            if (this.nextIndices[i] < this.lists[i].size()) {
                nextSite = Math.min(nextSite, this.lists[i].getSite(this.nextIndices[i]));
            }
        }

        boolean hasState = false;
        for (int i = 0; i < this.numLists; i++) {
            if (this.nextIndices[i] < this.lists[i].size() && this.lists[i].getSite(this.nextIndices[i]) == nextSite) {
                this.currentIndices[i] = this.nextIndices[i]++;

                // the changes of the node end in its state, those of a child start in it
                if (!hasState) {
                    this.currentState = this.hasParent && i == 0
                            ? this.lists[i].getTo(this.currentIndices[i])
                            : this.lists[i].getFrom(this.currentIndices[i]);
                    hasState = true;
                }
            } else {
                this.currentIndices[i] = -1;
            }
        }

        return nextSite;
    }

    /** Returns the current state of the node at the current site. */
    int getCurrentState() {
        return this.currentState;
    }

    /** Returns the state of the neighbour of the given list at the current site, which is the parent for the first list of a node with a parent. */
    int getState(int list) {
        int index = this.currentIndices[list];
        if (index < 0) {
            return this.currentState;
        }
        return this.hasParent && list == 0 ? this.lists[list].getFrom(index) : this.lists[list].getTo(index);
    }

}
