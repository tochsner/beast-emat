package esat.helper;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.datatype.DataType;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import esat.state.StateChanges;

import java.util.Arrays;

/**
 * The constraints of the alignment on the states of the tips. A site constrains a tip if its
 * code is compatible with some but not all states, which includes every unambiguous site.
 * Missing sites, which are compatible with every state, do not constrain the tip. The states
 * of a tip at its unconstrained and ambiguous sites are latent, and only need to be among
 * the compatible states. The tips keep their numbers and taxa when the tree changes.
 */
public final class TipCompatibility {

    private final int numStates;
    private final int everyStateMask;

    // per node: the sorted constrained sites, and the bit mask of the compatible states at each of them
    private final int[][] constrainedSitesOfNode;
    private final int[][] compatibleStatesOfNode;

    // per node: the sorted latent sites, split into the missing sites and the ambiguous sites with their bit masks
    private final int[][] missingSitesOfNode;
    private final int[][] ambiguousSitesOfNode;
    private final int[][] ambiguousStatesOfNode;

    // the alignment, the taxon of every tip node, and the bit mask of the compatible states of every code in the alignment
    private final Alignment alignment;
    private final int[] taxonIndexOfNode;
    private final int[] statesOfCode;

    public TipCompatibility(Alignment alignment, TreeInterface tree) {
        int numSites = alignment.getSiteCount();
        int numNodes = tree.getNodeCount();

        this.numStates = alignment.getMaxStateCount();
        if (this.numStates > Integer.SIZE - 1) {
            throw new IllegalArgumentException("Only data types with up to " + (Integer.SIZE - 1) + " states are supported.");
        }
        this.everyStateMask = (1 << this.numStates) - 1;

        this.constrainedSitesOfNode = new int[numNodes][];
        this.compatibleStatesOfNode = new int[numNodes][];
        this.missingSitesOfNode = new int[numNodes][];
        this.ambiguousSitesOfNode = new int[numNodes][];
        this.ambiguousStatesOfNode = new int[numNodes][];
        Arrays.fill(this.constrainedSitesOfNode, new int[0]);
        Arrays.fill(this.compatibleStatesOfNode, new int[0]);
        Arrays.fill(this.missingSitesOfNode, new int[0]);
        Arrays.fill(this.ambiguousSitesOfNode, new int[0]);
        Arrays.fill(this.ambiguousStatesOfNode, new int[0]);

        this.alignment = alignment;
        this.statesOfCode = computeStatesOfCodes(alignment);
        this.taxonIndexOfNode = new int[numNodes];
        Arrays.fill(this.taxonIndexOfNode, -1);

        for (Node tip : tree.getExternalNodes()) {
            int taxonIndex = alignment.getTaxonIndex(tip.getID());
            if (taxonIndex < 0) {
                throw new IllegalArgumentException("Tip " + tip.getID() + " is not in the alignment.");
            }
            this.taxonIndexOfNode[tip.getNr()] = taxonIndex;

            int[] sites = new int[numSites];
            int[] masks = new int[numSites];
            int[] missingSites = new int[numSites];
            int[] ambiguousSites = new int[numSites];
            int[] ambiguousMasks = new int[numSites];
            int numConstrainedSites = 0;
            int numMissingSites = 0;
            int numAmbiguousSites = 0;
            for (int site = 0; site < numSites; site++) {
                int mask = this.statesOfCode[alignment.getPattern(taxonIndex, alignment.getPatternIndex(site))];

                if (mask == this.everyStateMask) {
                    missingSites[numMissingSites++] = site;
                    continue;
                }

                sites[numConstrainedSites] = site;
                masks[numConstrainedSites] = mask;
                numConstrainedSites++;

                if (Integer.bitCount(mask) > 1) {
                    ambiguousSites[numAmbiguousSites] = site;
                    ambiguousMasks[numAmbiguousSites] = mask;
                    numAmbiguousSites++;
                }
            }

            this.constrainedSitesOfNode[tip.getNr()] = Arrays.copyOf(sites, numConstrainedSites);
            this.compatibleStatesOfNode[tip.getNr()] = Arrays.copyOf(masks, numConstrainedSites);
            this.missingSitesOfNode[tip.getNr()] = Arrays.copyOf(missingSites, numMissingSites);
            this.ambiguousSitesOfNode[tip.getNr()] = Arrays.copyOf(ambiguousSites, numAmbiguousSites);
            this.ambiguousStatesOfNode[tip.getNr()] = Arrays.copyOf(ambiguousMasks, numAmbiguousSites);
        }
    }

    /** Returns the bit mask of the compatible states of every code that occurs in the alignment, indexed by the code. */
    private static int[] computeStatesOfCodes(Alignment alignment) {
        DataType dataType = alignment.getDataType();

        int maxCode = 0;
        for (int patternNr = 0; patternNr < alignment.getPatternCount(); patternNr++) {
            for (int taxonIndex = 0; taxonIndex < alignment.getTaxonCount(); taxonIndex++) {
                maxCode = Math.max(maxCode, alignment.getPattern(taxonIndex, patternNr));
            }
        }

        int[] statesOfCode = new int[maxCode + 1];
        for (int patternNr = 0; patternNr < alignment.getPatternCount(); patternNr++) {
            for (int taxonIndex = 0; taxonIndex < alignment.getTaxonCount(); taxonIndex++) {
                int code = alignment.getPattern(taxonIndex, patternNr);
                if (statesOfCode[code] == 0) {
                    for (int state : dataType.getStatesForCode(code)) {
                        statesOfCode[code] |= 1 << state;
                    }
                }
            }
        }
        return statesOfCode;
    }

    /**
     * Returns whether the states of the given node, given by its changes compared to the root
     * sequence, are compatible with its data at every site. Inner nodes are always compatible.
     * This merges the constrained sites with the changes, so it takes O(C + m).
     */
    public boolean isCompatible(Node node, StateChanges changes, int[] rootSequence) {
        int[] sites = this.constrainedSitesOfNode[node.getNr()];
        int[] masks = this.compatibleStatesOfNode[node.getNr()];

        int j = 0;
        for (int i = 0; i < sites.length; i++) {
            int site = sites[i];
            while (j < changes.size() && changes.getSite(j) < site) {
                j++;
            }

            int state = j < changes.size() && changes.getSite(j) == site ? changes.getState(j) : rootSequence[site];
            if ((masks[i] & (1 << state)) == 0) {
                return false;
            }
        }

        return true;
    }

    /** Returns the bit mask of the states of the given node at the given site that are compatible with its data. */
    public int getCompatibleStates(Node node, int site) {
        int[] sites = this.constrainedSitesOfNode[node.getNr()];
        int index = Arrays.binarySearch(sites, site);
        return index >= 0 ? this.compatibleStatesOfNode[node.getNr()][index] : this.everyStateMask;
    }

    /** Returns the bit mask of the states of the given tip that are compatible with its data at the sites of the given pattern. */
    public int getCompatibleStatesOfPattern(Node tip, int patternNr) {
        return this.statesOfCode[this.alignment.getPattern(this.taxonIndexOfNode[tip.getNr()], patternNr)];
    }

    /** Returns the sorted sites at which the data constrains the states of the given node. */
    public int[] getConstrainedSites(Node node) {
        return this.constrainedSitesOfNode[node.getNr()];
    }

    /** Returns the sorted sites at which the data is compatible with every state of the given node. */
    public int[] getMissingSites(Node node) {
        return this.missingSitesOfNode[node.getNr()];
    }

    /** Returns the sorted sites at which the data is compatible with more than one but not every state of the given node. */
    public int[] getAmbiguousSites(Node node) {
        return this.ambiguousSitesOfNode[node.getNr()];
    }

    /** Returns the bit masks of the compatible states at the ambiguous sites of the given node, in the order of the sites. */
    public int[] getAmbiguousStates(Node node) {
        return this.ambiguousStatesOfNode[node.getNr()];
    }

    /** Returns whether the state of the given node at the given site is latent, i.e. compatible with more than one state. */
    public boolean isLatent(Node node, int site) {
        return Arrays.binarySearch(this.missingSitesOfNode[node.getNr()], site) >= 0
                || Arrays.binarySearch(this.ambiguousSitesOfNode[node.getNr()], site) >= 0;
    }

    public int getNumStates() {
        return this.numStates;
    }

    public int getEveryStateMask() {
        return this.everyStateMask;
    }

}
