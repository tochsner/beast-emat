package esat.state;

import java.util.Arrays;

/**
 * The states of one node at the sites where it differs from its parent, given as the state of
 * the parent and the state of the node at each of them. The changes are sorted by site, so
 * that the change at a single site is found by binary search in O(log m), and all changes are
 * traversed in site order in O(m). The changes of the root are always empty. The changes are
 * immutable, so that they can be shared between the current and the stored state.
 */
public final class BranchChanges {

    private final int nodeNr;

    // the sites in strictly ascending order, and the state of the parent and of the node at each of them
    private final int[] sites;
    private final int[] fromStates;
    private final int[] toStates;

    private BranchChanges(int nodeNr, int[] sites, int[] fromStates, int[] toStates) {
        this.nodeNr = nodeNr;
        this.sites = sites;
        this.fromStates = fromStates;
        this.toStates = toStates;
    }

    /** Returns the changes of a node that equals its parent everywhere. */
    public static BranchChanges empty(int nodeNr) {
        return new BranchChanges(nodeNr, new int[0], new int[0], new int[0]);
    }

    /**
     * Returns the changes given as three parallel arrays of sites and states, which may be in
     * any order but must not contain a site twice. The arrays are copied.
     */
    public static BranchChanges of(int nodeNr, int[] sites, int[] fromStates, int[] toStates) {
        if (sites.length != fromStates.length || sites.length != toStates.length) {
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
        int[] sortedFromStates = new int[sites.length];
        int[] sortedToStates = new int[sites.length];
        for (int i = 0; i < keys.length; i++) {
            int index = (int) keys[i];
            sortedSites[i] = (int) (keys[i] >>> 32);
            sortedFromStates[i] = fromStates[index];
            sortedToStates[i] = toStates[index];
        }

        return ofSorted(nodeNr, sortedSites, sortedFromStates, sortedToStates, sortedSites.length);
    }

    /**
     * Returns the changes given by the first size entries of three parallel arrays, whose
     * sites must be strictly ascending and whose states must differ. The entries are copied, so
     * that the arrays can be reused.
     */
    public static BranchChanges ofSorted(int nodeNr, int[] sites, int[] fromStates, int[] toStates, int size) {
        for (int i = 0; i < size; i++) {
            if (i > 0 && sites[i] <= sites[i - 1]) {
                throw new IllegalArgumentException("Sites must be strictly ascending.");
            }
            if (fromStates[i] == toStates[i]) {
                throw new IllegalArgumentException("Change at site " + sites[i] + " does not differ from the parent.");
            }
        }
        return new BranchChanges(nodeNr, Arrays.copyOf(sites, size), Arrays.copyOf(fromStates, size), Arrays.copyOf(toStates, size));
    }

    /** Returns the changes of a node with the given full sequence compared to the full sequence of its parent. */
    public static BranchChanges fromSequences(int nodeNr, int[] parentSequence, int[] sequence) {
        if (sequence.length != parentSequence.length) {
            throw new IllegalArgumentException("Expected a sequence as long as the sequence of the parent.");
        }

        int[] sites = new int[sequence.length];
        int[] fromStates = new int[sequence.length];
        int[] toStates = new int[sequence.length];
        int size = 0;
        for (int site = 0; site < sequence.length; site++) {
            if (sequence[site] != parentSequence[site]) {
                sites[size] = site;
                fromStates[size] = parentSequence[site];
                toStates[size] = sequence[site];
                size++;
            }
        }

        return new BranchChanges(nodeNr, Arrays.copyOf(sites, size), Arrays.copyOf(fromStates, size), Arrays.copyOf(toStates, size));
    }

    public int getNodeNr() {
        return this.nodeNr;
    }

    /** Returns the number of sites at which the node differs from its parent. */
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

    /** Returns the state of the parent at the change with the given index. */
    public int getFrom(int index) {
        return this.fromStates[index];
    }

    /** Returns the state of the node at the change with the given index. */
    public int getTo(int index) {
        return this.toStates[index];
    }

    /** Returns the index of the change at the given site, or a negative value if the node equals its parent there. */
    public int indexOfSite(int site) {
        return Arrays.binarySearch(this.sites, site);
    }

    /** Returns whether the given changes list the same sites and states as these. */
    public boolean hasSameEntries(BranchChanges other) {
        return Arrays.equals(this.sites, other.sites)
                && Arrays.equals(this.fromStates, other.fromStates)
                && Arrays.equals(this.toStates, other.toStates);
    }

}
