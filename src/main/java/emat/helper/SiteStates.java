package emat.helper;

import java.util.Arrays;

/**
 * A reusable map from sites to their states at a fixed number of points of the tree, the
 * columns. Entries are kept in slots in the order they were added, and a lookup table over
 * all sites finds the slot of a site in constant time without boxing. Clearing only resets
 * the added sites, so its cost does not depend on the genome length.
 */
public final class SiteStates {

    private static final int INITIAL_CAPACITY = 16;

    private final int numColumns;

    // the slot of every site, or -1 if the site has no entry
    private final int[] slotOfSite;

    private int[] sites = new int[INITIAL_CAPACITY];
    private int[] states;
    private int size = 0;

    public SiteStates(int numSites, int numColumns) {
        this.numColumns = numColumns;
        this.slotOfSite = new int[numSites];
        Arrays.fill(this.slotOfSite, -1);
        this.states = new int[INITIAL_CAPACITY * numColumns];
    }

    public int getNumColumns() {
        return this.numColumns;
    }

    /** Returns the number of sites with an entry, which are stored in the slots below it. */
    public int getSize() {
        return this.size;
    }

    public int getSite(int slot) {
        return this.sites[slot];
    }

    public int getState(int slot, int column) {
        return this.states[slot * this.numColumns + column];
    }

    public void setState(int slot, int column, int state) {
        this.states[slot * this.numColumns + column] = state;
    }

    /** Sets the state of every column of the given slot to the given state. */
    public void setAllStates(int slot, int state) {
        Arrays.fill(this.states, slot * this.numColumns, (slot + 1) * this.numColumns, state);
    }

    /** Returns the slot of the given site, or -1 if it has no entry. */
    public int getSlot(int site) {
        return this.slotOfSite[site];
    }

    public boolean containsSite(int site) {
        return this.slotOfSite[site] >= 0;
    }

    /** Checks whether the states of the given slot agree in all columns below the given one. */
    public boolean isAgreeingBelow(int slot, int column) {
        int state = this.getState(slot, 0);
        for (int i = 1; i < column; i++) {
            if (this.getState(slot, i) != state) {
                return false;
            }
        }
        return true;
    }

    /**
     * Adds an entry for the given site, which must not have one yet, and returns its slot.
     * Its states must be set afterwards.
     */
    public int addSite(int site) {
        if (this.size == this.sites.length) {
            this.grow();
        }

        int slot = this.size++;
        this.sites[slot] = site;
        this.slotOfSite[site] = slot;
        return slot;
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
        this.states = Arrays.copyOf(this.states, capacity * this.numColumns);
    }

}
