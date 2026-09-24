package emat.helper;

import java.util.Arrays;

/**
 * A reusable map from sites to their states at two points of the tree, the start and the
 * end. Entries are kept in slots in the order they were added, and a lookup table over all
 * sites finds the slot of a site in constant time without boxing. Clearing only resets the
 * added sites, so its cost does not depend on the genome length.
 */
public final class SiteChanges {

    private static final int INITIAL_CAPACITY = 16;

    // the slot of every site, or -1 if the site has no entry
    private final int[] slotOfSite;

    private int[] sites = new int[INITIAL_CAPACITY];
    private int[] startStates = new int[INITIAL_CAPACITY];
    private int[] endStates = new int[INITIAL_CAPACITY];
    private int size = 0;

    public SiteChanges(int numSites) {
        this.slotOfSite = new int[numSites];
        Arrays.fill(this.slotOfSite, -1);
    }

    /** Returns the number of sites with an entry, which are stored in the slots below it. */
    public int getSize() {
        return this.size;
    }

    public int getSite(int slot) {
        return this.sites[slot];
    }

    public int getStartState(int slot) {
        return this.startStates[slot];
    }

    public int getEndState(int slot) {
        return this.endStates[slot];
    }

    /** Returns the slot of the given site, or -1 if it has no entry. */
    public int getSlot(int site) {
        return this.slotOfSite[site];
    }

    public boolean containsSite(int site) {
        return this.slotOfSite[site] >= 0;
    }

    /** Adds an entry for the given site, which must not have one yet, and returns its slot. */
    public int addSite(int site, int startState, int endState) {
        if (this.size == this.sites.length) {
            this.grow();
        }

        int slot = this.size++;
        this.sites[slot] = site;
        this.startStates[slot] = startState;
        this.endStates[slot] = endState;
        this.slotOfSite[site] = slot;
        return slot;
    }

    public void setStartState(int slot, int state) {
        this.startStates[slot] = state;
    }

    /** Removes all entries. */
    public void clear() {
        for (int slot = 0; slot < this.size; slot++) {
            this.slotOfSite[this.sites[slot]] = -1;
        }
        this.size = 0;
    }

    private void grow() {
        int capacity = 2 * this.sites.length;
        this.sites = Arrays.copyOf(this.sites, capacity);
        this.startStates = Arrays.copyOf(this.startStates, capacity);
        this.endStates = Arrays.copyOf(this.endStates, capacity);
    }

}
