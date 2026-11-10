package esat.stochasticmapping;

import esat.state.StateChanges;

/**
 * A cursor over the union of the sites of several change lists, in ascending order. At every
 * site of the union, it gives the state of each list, which is the state of its change if it
 * lists the site and the reference state otherwise. A step costs O(k) for k lists, and the
 * arrays are reused between merges.
 */
final class ChangeListMerge {

    private StateChanges[] lists = new StateChanges[0];
    private int numLists = 0;
    private int[] reference;

    // per list: the index of its next unvisited change, and the index of its change at the current site or -1
    private int[] nextIndices = new int[0];
    private int[] currentIndices = new int[0];

    private int site = -1;

    /** Starts a merge of the first given number of lists, whose unlisted sites take the state of the given reference. */
    void reset(StateChanges[] lists, int numLists, int[] reference) {
        if (this.lists.length < numLists) {
            this.lists = new StateChanges[numLists];
            this.nextIndices = new int[numLists];
            this.currentIndices = new int[numLists];
        }

        System.arraycopy(lists, 0, this.lists, 0, numLists);
        this.numLists = numLists;
        this.reference = reference;
        for (int i = 0; i < numLists; i++) {
            this.nextIndices[i] = 0;
            this.currentIndices[i] = -1;
        }
        this.site = -1;
    }

    /** Moves to the next site of the union and returns it, or returns Integer.MAX_VALUE if no site is left. */
    int nextSite() {
        int nextSite = Integer.MAX_VALUE;
        for (int i = 0; i < this.numLists; i++) {
            if (this.nextIndices[i] < this.lists[i].size()) {
                nextSite = Math.min(nextSite, this.lists[i].getSite(this.nextIndices[i]));
            }
        }

        for (int i = 0; i < this.numLists; i++) {
            if (this.nextIndices[i] < this.lists[i].size() && this.lists[i].getSite(this.nextIndices[i]) == nextSite) {
                this.currentIndices[i] = this.nextIndices[i]++;
            } else {
                this.currentIndices[i] = -1;
            }
        }

        this.site = nextSite;
        return nextSite;
    }

    /** Returns the state of the given list at the current site. */
    int getState(int list) {
        int index = this.currentIndices[list];
        return index >= 0 ? this.lists[list].getState(index) : this.reference[this.site];
    }

}
