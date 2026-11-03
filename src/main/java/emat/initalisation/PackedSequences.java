package emat.initalisation;

import beast.base.evolution.alignment.Alignment;

import java.util.ArrayList;
import java.util.List;

/**
 * The variable sites of the tip sequences, packed into bit sets, so that counting the
 * differences between two tips takes a few bit operations per 64 sites. A site is variable if
 * at least two tips have different valid states there, and a state is valid if it is below the
 * number of states, so missing and ambiguous characters are left out. The sites are expanded
 * from the patterns of the alignment by their weights. For every tip and word of 64 sites, the
 * bits of the states are stored next to each other, at index word * numStates + state.
 */
public class PackedSequences {

    private final int numTips;
    private final int numStates;
    private final int numSites;
    private final int numVariableSites;
    private final int numWords;

    // per tip, the variable sites in each state, and the variable sites with a valid state
    private final long[][] stateBits;
    private final long[][] validBits;

    /**
     * Packs the sequences of the given taxa, in the given order, so that tip i is the taxon
     * with index i in the given names.
     */
    public PackedSequences(Alignment alignment, String[] taxaNames) {
        this.numTips = taxaNames.length;
        this.numStates = alignment.getMaxStateCount();
        this.numSites = alignment.getSiteCount();
        if (this.numStates > Long.SIZE) {
            throw new IllegalArgumentException("At most " + Long.SIZE + " states are supported.");
        }

        int[] alignmentIndices = this.computeAlignmentIndices(alignment, taxaNames);
        List<Integer> variablePatterns = this.collectVariablePatterns(alignment, alignmentIndices);
        this.numVariableSites = variablePatterns.stream().mapToInt(alignment::getPatternWeight).sum();
        this.numWords = (this.numVariableSites + Long.SIZE - 1) / Long.SIZE;

        this.stateBits = new long[this.numTips][this.numWords * this.numStates];
        this.validBits = new long[this.numTips][this.numWords];
        this.packSites(alignment, alignmentIndices, variablePatterns);
    }

    /** Returns the index of every taxon in the alignment, in the order of the given names. */
    private int[] computeAlignmentIndices(Alignment alignment, String[] taxaNames) {
        List<String> alignmentTaxa = alignment.getTaxaNames();
        int[] alignmentIndices = new int[taxaNames.length];
        for (int i = 0; i < taxaNames.length; i++) {
            alignmentIndices[i] = alignmentTaxa.indexOf(taxaNames[i]);
            if (alignmentIndices[i] < 0) {
                throw new IllegalArgumentException("Taxon " + taxaNames[i] + " is not in the alignment.");
            }
        }
        return alignmentIndices;
    }

    /** Returns the patterns where at least two of the given taxa have different valid states. */
    private List<Integer> collectVariablePatterns(Alignment alignment, int[] alignmentIndices) {
        List<Integer> variablePatterns = new ArrayList<>();
        for (int pattern = 0; pattern < alignment.getPatternCount(); pattern++) {
            long seenStates = 0L;
            for (int taxon : alignmentIndices) {
                int state = alignment.getPattern(taxon, pattern);
                if (state < this.numStates) {
                    seenStates |= 1L << state;
                }
            }
            if (Long.bitCount(seenStates) > 1) {
                variablePatterns.add(pattern);
            }
        }
        return variablePatterns;
    }

    /** Sets the bits of every variable site, repeating each pattern by its weight. */
    private void packSites(Alignment alignment, int[] alignmentIndices, List<Integer> variablePatterns) {
        int site = 0;
        for (int pattern : variablePatterns) {
            for (int copy = 0; copy < alignment.getPatternWeight(pattern); copy++) {
                int word = site / Long.SIZE;
                long bit = 1L << (site % Long.SIZE);

                for (int tip = 0; tip < this.numTips; tip++) {
                    int state = alignment.getPattern(alignmentIndices[tip], pattern);
                    if (state < this.numStates) {
                        this.stateBits[tip][word * this.numStates + state] |= bit;
                        this.validBits[tip][word] |= bit;
                    }
                }
                site++;
            }
        }
    }

    /**
     * Counts the variable sites where both tips have a valid state and the states differ. The
     * count only grows from word to word, so the counting stops as soon as it reaches the given
     * bound, and then returns a number of at least the bound.
     */
    public int countDifferences(int tip1, int tip2, int bound) {
        long[] states1 = this.stateBits[tip1];
        long[] states2 = this.stateBits[tip2];
        long[] valid1 = this.validBits[tip1];
        long[] valid2 = this.validBits[tip2];

        int numDifferences = 0;
        for (int word = 0; word < this.numWords && numDifferences < bound; word++) {
            // the states are exclusive, so a site agrees if any state bit is set in both tips
            long agreeing = 0L;
            int offset = word * this.numStates;
            for (int state = 0; state < this.numStates; state++) {
                agreeing |= states1[offset + state] & states2[offset + state];
            }
            numDifferences += Long.bitCount(valid1[word] & valid2[word] & ~agreeing);
        }
        return numDifferences;
    }

    /**
     * Returns the number of changes on the branch above every node of a Fitch parsimony
     * reconstruction of the variable sites on the given topology, whose tips are the packed
     * sequences. A tip without a valid state at a site may have any state there. The root has
     * no branch and no changes.
     */
    public int[] countParsimonyChanges(BinaryTopology topology) {
        long[][] stateSets = this.computeFitchStateSets(topology);
        long[][] states = new long[topology.getNumNodes()][this.numWords * this.numStates];
        int[] numChanges = new int[topology.getNumNodes()];

        for (int node : topology.getPreorder()) {
            int parent = topology.getParent(node);
            if (parent < 0) {
                // without parent states, the root takes the lowest state of its set everywhere
                this.chooseStates(stateSets[node], new long[this.numWords * this.numStates], states[node]);
            } else {
                numChanges[node] = this.chooseStates(stateSets[node], states[parent], states[node]);
            }
        }
        return numChanges;
    }

    /**
     * Returns the Fitch state sets of every node from the tips up: the intersection of the sets
     * of the two children where it is not empty, and their union elsewhere. A tip allows every
     * state where it has no valid state, which includes the bits beyond the last site, so these
     * never count as changes.
     */
    private long[][] computeFitchStateSets(BinaryTopology topology) {
        long[][] stateSets = new long[topology.getNumNodes()][this.numWords * this.numStates];

        for (int node : topology.getPostorder()) {
            int[] children = topology.getChildren(node);
            long[] stateSet = stateSets[node];
            for (int word = 0; word < this.numWords; word++) {
                int offset = word * this.numStates;
                if (children.length == 0) {
                    long invalid = ~this.validBits[node][word];
                    for (int state = 0; state < this.numStates; state++) {
                        stateSet[offset + state] = this.stateBits[node][offset + state] | invalid;
                    }
                    continue;
                }

                long[] left = stateSets[children[0]];
                long[] right = stateSets[children[1]];
                long shared = 0L;
                for (int state = 0; state < this.numStates; state++) {
                    shared |= left[offset + state] & right[offset + state];
                }
                for (int state = 0; state < this.numStates; state++) {
                    long intersection = left[offset + state] & right[offset + state];
                    long union = left[offset + state] | right[offset + state];
                    stateSet[offset + state] = intersection | (union & ~shared);
                }
            }
        }
        return stateSets;
    }

    /**
     * Sets one state per site of a node: the state of the parent where the state set allows it,
     * and the lowest state of the state set elsewhere. Returns the number of sites where the
     * state differs from the state of the parent.
     */
    private int chooseStates(long[] stateSet, long[] parentStates, long[] states) {
        int numChanges = 0;
        for (int word = 0; word < this.numWords; word++) {
            int offset = word * this.numStates;
            long kept = 0L;
            for (int state = 0; state < this.numStates; state++) {
                kept |= stateSet[offset + state] & parentStates[offset + state];
            }

            long open = ~kept;
            for (int state = 0; state < this.numStates; state++) {
                long chosen = stateSet[offset + state] & open;
                states[offset + state] = (parentStates[offset + state] & kept) | chosen;
                open &= ~chosen;
            }
            numChanges += Long.bitCount(~kept);
        }
        return numChanges;
    }

    /** Returns the number of all sites of the alignment, including the invariant ones. */
    public int getNumSites() {
        return this.numSites;
    }

    public int getNumVariableSites() {
        return this.numVariableSites;
    }

}
