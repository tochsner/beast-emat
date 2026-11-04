package emat.helper;

import java.util.Arrays;

/**
 * A reusable map from sites to their states at two points of the tree, the start and the
 * end. Entries are kept in slots in the order they were added, and a lookup table over all
 * sites finds the slot of a site in constant time. The slots are stored in primitive arrays
 * that grow as needed, so adding an entry does not allocate. The lookup table tags every
 * slot with the generation of the map it was added in, so clearing only starts a new
 * generation and takes constant time.
 */
public final class SiteChanges {

    private static final int INITIAL_CAPACITY = 16;

    // per site, the generation in the upper and the slot in the lower 32 bits. An entry is only valid in the current
    // generation, which starts at one, so that the zeros the table starts with are invalid.
    private final long[] slotOfSite;
    private int generation = 1;

    // the site and both states of every slot
    private int[] sites = new int[INITIAL_CAPACITY];
    private int[] startStates = new int[INITIAL_CAPACITY];
    private int[] endStates = new int[INITIAL_CAPACITY];
    private int size = 0;

    public SiteChanges(int numSites) {
        this.slotOfSite = new long[numSites];
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
        long entry = this.slotOfSite[site];
        return (int) (entry >>> 32) == this.generation ? (int) entry : -1;
    }

    public boolean containsSite(int site) {
        return (int) (this.slotOfSite[site] >>> 32) == this.generation;
    }

    /** Adds an entry for the given site, which must not have one yet, and returns its slot. */
    public int addSite(int site, int startState, int endState) {
        if (this.size == this.sites.length) {
            this.growCapacity();
        }

        int slot = this.size++;
        this.sites[slot] = site;
        this.startStates[slot] = startState;
        this.endStates[slot] = endState;
        this.slotOfSite[site] = ((long) this.generation << 32) | slot;
        return slot;
    }

    /** Doubles the number of slots, which never needs to exceed the number of sites. */
    private void growCapacity() {
        int capacity = Math.min(2 * this.sites.length, this.slotOfSite.length);
        this.sites = Arrays.copyOf(this.sites, capacity);
        this.startStates = Arrays.copyOf(this.startStates, capacity);
        this.endStates = Arrays.copyOf(this.endStates, capacity);
    }

    public void setStartState(int slot, int state) {
        this.startStates[slot] = state;
    }

    /** Removes all entries by starting a new generation, which only needs resetting the table once the counter runs out. */
    public void clear() {
        if (this.generation == Integer.MAX_VALUE) {
            Arrays.fill(this.slotOfSite, 0L);
            this.generation = 0;
        }
        this.generation++;
        this.size = 0;
    }

}
