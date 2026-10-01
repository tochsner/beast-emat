package emat.helper;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * A reusable map from sites to their states at a fixed number of points of the tree, the
 * columns. Entries are kept in slots in the order they were added, and a lookup table over
 * all sites finds the slot of a site in constant time without boxing. Clearing only resets
 * the added sites, so its cost does not depend on the genome length.
 */
public final class SiteStates {

    private record Entry(int site, int[] states) {}

    private final int numColumns;

    // the slot of every site, or -1 if the site has no entry
    private final int[] slotOfSite;

    private final List<Entry> entries = new ArrayList<>();

    public SiteStates(int numSites, int numColumns) {
        this.numColumns = numColumns;
        this.slotOfSite = new int[numSites];
        Arrays.fill(this.slotOfSite, -1);
    }

    public int getNumColumns() {
        return this.numColumns;
    }

    /** Returns the number of sites with an entry, which are stored in the slots below it. */
    public int getSize() {
        return this.entries.size();
    }

    public int getSite(int slot) {
        return this.entries.get(slot).site();
    }

    public int getState(int slot, int column) {
        return this.entries.get(slot).states()[column];
    }

    public void setState(int slot, int column, int state) {
        this.entries.get(slot).states()[column] = state;
    }

    /** Sets the state of every column of the given slot to the given state. */
    public void setAllStates(int slot, int state) {
        Arrays.fill(this.entries.get(slot).states(), state);
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
        int[] states = this.entries.get(slot).states();
        for (int i = 1; i < column; i++) {
            if (states[i] != states[0]) {
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
        int slot = this.entries.size();
        this.entries.add(new Entry(site, new int[this.numColumns]));
        this.slotOfSite[site] = slot;
        return slot;
    }

    /** Removes all entries. */
    public void clear() {
        for (Entry entry : this.entries) {
            this.slotOfSite[entry.site()] = -1;
        }
        this.entries.clear();
    }

}
