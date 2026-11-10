package esat.stochasticmapping;

import java.util.Arrays;

/**
 * New states of one node at sorted sites, where each state differs from the current state of
 * the node. This is what a sampler of the states of a node returns, and what
 * NodeStates.setStates applies.
 */
public record StateEdit(int nodeNr, int[] sites, int[] states) {

    /**
     * Returns the edit given by two parallel arrays of sites and states, which may be in any
     * order but must not contain a site twice. The arrays are copied.
     */
    public static StateEdit of(int nodeNr, int[] sites, int[] states) {
        if (sites.length != states.length) {
            throw new IllegalArgumentException("Expected as many states as sites.");
        }

        // sort the entries by site without boxing, packing the site and the entry index into one long

        long[] keys = new long[sites.length];
        for (int i = 0; i < sites.length; i++) {
            keys[i] = ((long) sites[i] << 32) | i;
        }
        Arrays.sort(keys);

        int[] sortedSites = new int[sites.length];
        int[] sortedStates = new int[sites.length];
        for (int i = 0; i < keys.length; i++) {
            sortedSites[i] = (int) (keys[i] >>> 32);
            sortedStates[i] = states[(int) keys[i]];
        }
        return new StateEdit(nodeNr, sortedSites, sortedStates);
    }

    /** Returns the edit given by the first size entries of two parallel arrays, whose sites are ascending. The entries are copied. */
    public static StateEdit ofSorted(int nodeNr, int[] sites, int[] states, int size) {
        return new StateEdit(nodeNr, Arrays.copyOf(sites, size), Arrays.copyOf(states, size));
    }

    public int size() {
        return this.sites.length;
    }

}
