package emat.helper;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.datatype.DataType;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import emat.state.Mutations;

import java.util.Arrays;
import java.util.BitSet;

/**
 * The sites where the data of every tip is missing, i.e. compatible with every state, and
 * the sites where it is partially ambiguous, i.e. compatible with several but not all
 * states, together with these states. Moves that change the placement of a tip may
 * resample its states at the missing sites, and among the compatible states at the
 * partially ambiguous sites. Inner nodes have neither.
 */
public final class MissingSites {

    // per node: the sorted missing sites, and the same sites as a set
    private final int[][] sitesOfNode;
    private final BitSet[] siteSetOfNode;

    // per node: the sorted partially ambiguous sites, the same sites as a set, and the bit mask of the compatible states at each of them
    private final int[][] ambiguousSitesOfNode;
    private final BitSet[] ambiguousSiteSetOfNode;
    private final int[][] compatibleStatesOfNode;

    public MissingSites(Mutations mutations, Tree tree) {
        Alignment alignment = mutations.getAlignment();
        DataType dataType = alignment.getDataType();
        int numStates = alignment.getMaxStateCount();
        int numSites = mutations.getReferenceSequence().length;
        int numNodes = tree.getNodeCount();

        this.sitesOfNode = new int[numNodes][];
        this.siteSetOfNode = new BitSet[numNodes];
        this.ambiguousSitesOfNode = new int[numNodes][];
        this.ambiguousSiteSetOfNode = new BitSet[numNodes];
        this.compatibleStatesOfNode = new int[numNodes][];
        Arrays.fill(this.sitesOfNode, new int[0]);
        Arrays.fill(this.siteSetOfNode, new BitSet());
        Arrays.fill(this.ambiguousSitesOfNode, new int[0]);
        Arrays.fill(this.ambiguousSiteSetOfNode, new BitSet());
        Arrays.fill(this.compatibleStatesOfNode, new int[0]);

        // the tips keep their numbers and taxa when the tree changes
        for (Node tip : tree.getExternalNodes()) {
            int taxonIndex = alignment.getTaxonIndex(tip.getID());

            BitSet missingSites = new BitSet(numSites);
            BitSet ambiguousSites = new BitSet(numSites);
            int[] compatibleStates = new int[numSites];
            for (int site = 0; site < numSites; site++) {
                int code = alignment.getPattern(taxonIndex, alignment.getPatternIndex(site));
                int[] states = dataType.getStatesForCode(code);
                if (states.length == numStates) {
                    missingSites.set(site);
                } else if (states.length > 1) {
                    ambiguousSites.set(site);
                    for (int state : states) {
                        compatibleStates[site] |= 1 << state;
                    }
                }
            }

            int[] sortedAmbiguousSites = ambiguousSites.stream().toArray();
            this.sitesOfNode[tip.getNr()] = missingSites.stream().toArray();
            this.siteSetOfNode[tip.getNr()] = missingSites;
            this.ambiguousSitesOfNode[tip.getNr()] = sortedAmbiguousSites;
            this.ambiguousSiteSetOfNode[tip.getNr()] = ambiguousSites;
            this.compatibleStatesOfNode[tip.getNr()] = Arrays.stream(sortedAmbiguousSites).map(site -> compatibleStates[site]).toArray();
        }
    }

    /** Returns the sorted missing sites of the given node. */
    public int[] getSites(Node node) {
        return this.sitesOfNode[node.getNr()];
    }

    /** Returns the missing sites of the given node as a set. */
    public BitSet getSiteSet(Node node) {
        return this.siteSetOfNode[node.getNr()];
    }

    /** Returns the sorted partially ambiguous sites of the given node. */
    public int[] getAmbiguousSites(Node node) {
        return this.ambiguousSitesOfNode[node.getNr()];
    }

    /** Returns the partially ambiguous sites of the given node as a set. */
    public BitSet getAmbiguousSiteSet(Node node) {
        return this.ambiguousSiteSetOfNode[node.getNr()];
    }

    /** Returns the bit masks of the compatible states at the partially ambiguous sites of the given node, in the order of getAmbiguousSites. */
    public int[] getCompatibleStates(Node node) {
        return this.compatibleStatesOfNode[node.getNr()];
    }

}
