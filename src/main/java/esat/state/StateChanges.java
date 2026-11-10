package esat.state;

import java.util.Arrays;

/**
 * The states of one node at the sites where they differ from the root sequence. The changes
 * are sorted by site, so that the state of a single site is found by binary search in
 * O(log m), and all changes are traversed in site order in O(m). Two nodes can therefore be
 * compared by merging their changes. The changes are immutable, so that they can be shared
 * between the current and the stored state.
 */
public final class StateChanges {

    private final int nodeNr;

    // the sites in strictly ascending order, and the state of the node at each of them
    private final int[] sites;
    private final int[] states;

    private StateChanges(int nodeNr, int[] sites, int[] states) {
        this.nodeNr = nodeNr;
        this.sites = sites;
        this.states = states;
    }

    /** Returns the changes of a node that equals the root sequence everywhere. */
    public static StateChanges empty(int nodeNr) {
        return new StateChanges(nodeNr, new int[0], new int[0]);
    }

    /**
     * Returns the changes given as two parallel arrays of sites and states, which may be in any
     * order but must not contain a site twice. The arrays are copied.
     */
    public static StateChanges of(int nodeNr, int[] sites, int[] states) {
        if (sites.length != states.length) {
            throw new IllegalArgumentException("Expected as many states as sites.");
        }

        // sort the entries by site without boxing, packing the site and the entry index into one long

        long[] keys = new long[sites.length];
        for (int i = 0; i < sites.length; i++) {
            if (sites[i] < 0) {
                throw new IllegalArgumentException("Sites must not be negative.");
            }
            keys[i] = ((long) sites[i] << 32) | i;
        }
        Arrays.sort(keys);

        int[] sortedSites = new int[sites.length];
        int[] sortedStates = new int[sites.length];
        for (int i = 0; i < keys.length; i++) {
            sortedSites[i] = (int) (keys[i] >>> 32);
            sortedStates[i] = states[(int) keys[i]];
        }

        return ofSorted(nodeNr, sortedSites, sortedStates, sortedSites.length);
    }

    /**
     * Returns the changes given by the first size entries of two parallel arrays, whose sites
     * must be strictly ascending. The entries are copied, so that the arrays can be reused.
     */
    public static StateChanges ofSorted(int nodeNr, int[] sites, int[] states, int size) {
        for (int i = 1; i < size; i++) {
            if (sites[i] <= sites[i - 1]) {
                throw new IllegalArgumentException("Sites must be strictly ascending.");
            }
        }
        return new StateChanges(nodeNr, Arrays.copyOf(sites, size), Arrays.copyOf(states, size));
    }

    /** Returns the changes of a node with the given full sequence compared to the given root sequence. */
    public static StateChanges fromSequence(int nodeNr, int[] sequence, int[] rootSequence) {
        if (sequence.length != rootSequence.length) {
            throw new IllegalArgumentException("Expected a sequence as long as the root sequence.");
        }

        int[] sites = new int[sequence.length];
        int[] states = new int[sequence.length];
        int size = 0;
        for (int site = 0; site < sequence.length; site++) {
            if (sequence[site] != rootSequence[site]) {
                sites[size] = site;
                states[size] = sequence[site];
                size++;
            }
        }

        return new StateChanges(nodeNr, Arrays.copyOf(sites, size), Arrays.copyOf(states, size));
    }

    public int getNodeNr() {
        return this.nodeNr;
    }

    /** Returns the number of sites at which the node differs from the root sequence. */
    public int size() {
        return this.sites.length;
    }

    public boolean isEmpty() {
        return this.sites.length == 0;
    }

    /** Returns the site of the change at the given index, in ascending order of sites. */
    public int getSite(int index) {
        return this.sites[index];
    }

    /** Returns the state of the node at the change with the given index. */
    public int getState(int index) {
        return this.states[index];
    }

    /** Returns the index of the change at the given site, or a negative value if the site does not change. */
    public int indexOfSite(int site) {
        return Arrays.binarySearch(this.sites, site);
    }

    /** Returns the state of the node at the given site, which is the given root state if the site does not change. */
    public int getStateOrDefault(int site, int rootState) {
        int index = Arrays.binarySearch(this.sites, site);
        return index >= 0 ? this.states[index] : rootState;
    }

}
