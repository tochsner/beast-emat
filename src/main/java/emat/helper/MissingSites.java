package emat.helper;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.datatype.DataType;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import emat.state.Mutations;

import java.util.Arrays;
import java.util.BitSet;

/**
 * The sites where the data of every tip is missing, i.e. compatible with every state. Moves
 * that change the placement of a tip may resample its states at these sites. Partially
 * ambiguous characters are not included, so their states stay fixed. Inner nodes have no
 * missing sites.
 */
public final class MissingSites {

    // per node: the sorted missing sites, and the same sites as a set
    private final int[][] sitesOfNode;
    private final BitSet[] siteSetOfNode;

    public MissingSites(Mutations mutations, Tree tree) {
        Alignment alignment = mutations.getAlignment();
        DataType dataType = alignment.getDataType();
        int numStates = alignment.getMaxStateCount();
        int numSites = mutations.getReferenceSequence().length;
        int numNodes = tree.getNodeCount();

        this.sitesOfNode = new int[numNodes][];
        this.siteSetOfNode = new BitSet[numNodes];
        Arrays.fill(this.sitesOfNode, new int[0]);
        Arrays.fill(this.siteSetOfNode, new BitSet());

        // the tips keep their numbers and taxa when the tree changes
        for (Node tip : tree.getExternalNodes()) {
            int taxonIndex = alignment.getTaxonIndex(tip.getID());

            BitSet missingSites = new BitSet(numSites);
            for (int site = 0; site < numSites; site++) {
                int code = alignment.getPattern(taxonIndex, alignment.getPatternIndex(site));
                if (dataType.getStatesForCode(code).length == numStates) {
                    missingSites.set(site);
                }
            }

            this.sitesOfNode[tip.getNr()] = missingSites.stream().toArray();
            this.siteSetOfNode[tip.getNr()] = missingSites;
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

}
