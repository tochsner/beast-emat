package emat.helper;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.datatype.DataType;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.spec.evolution.tree.ClusterTree;
import beast.base.util.Randomizer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Runs the upward pass of Fitch parsimony for every pattern of an alignment on a tree. It
 * provides the Fitch state sets of every node, as bit masks over the states, and the
 * parsimony score of every pattern and site, i.e. the minimum number of changes needed to
 * explain the tip data on the tree.
 */
public class FitchParsimony {

    final Alignment alignment;
    final TreeInterface tree;

    // the Fitch state sets per node and pattern, as bit masks over the states
    final long[][] stateSets;

    // the parsimony score per pattern
    final int[] patternScores;

    public FitchParsimony(Alignment alignment, TreeInterface tree) {
        if (alignment.getMaxStateCount() > Long.SIZE) {
            throw new IllegalArgumentException("Fitch parsimony supports at most " + Long.SIZE + " states.");
        }

        this.alignment = alignment;
        this.tree = tree;
        this.stateSets = new long[tree.getNodeCount()][alignment.getPatternCount()];
        this.patternScores = new int[alignment.getPatternCount()];

        this.computeStateSets();
    }

    /** Runs Fitch parsimony for the given alignment on its UPGMA tree. */
    public static FitchParsimony computeOnUpgmaTree(Alignment alignment) {
        return new FitchParsimony(alignment, buildUpgmaTree(alignment));
    }

    /**
     * Runs Fitch parsimony on the UPGMA tree of a uniformly random subsample of at most the
     * given number of sequences of the alignment. The sites are kept, so the site scores
     * refer to the sites of the full alignment.
     */
    public static FitchParsimony computeOnUpgmaTree(Alignment alignment, int maxNumSamples) {
        if (maxNumSamples < 2) {
            throw new IllegalArgumentException("At least two samples are needed to build a UPGMA tree.");
        }

        if (alignment.getTaxonCount() <= maxNumSamples) {
            return computeOnUpgmaTree(alignment);
        }

        return computeOnUpgmaTree(subsampleAlignment(alignment, maxNumSamples));
    }

    /** Returns an alignment of a uniformly random subset of the given number of sequences. */
    public static Alignment subsampleAlignment(Alignment alignment, int numSamples) {
        List<String> taxa = new ArrayList<>(alignment.getTaxaNames());

        // partial Fisher-Yates shuffle, so that the first entries are a uniform random subset

        for (int i = 0; i < numSamples; i++) {
            int j = i + Randomizer.nextInt(taxa.size() - i);
            Collections.swap(taxa, i, j);
        }

        List<Sequence> sequences = new ArrayList<>();
        for (String taxon : taxa.subList(0, numSamples)) {
            sequences.add(new Sequence(taxon, alignment.getSequenceAsString(taxon)));
        }

        Alignment subsample = new Alignment();
        subsample.initByName("sequence", sequences, "userDataType", alignment.getDataType());
        return subsample;
    }

    /** Builds the UPGMA tree of the given alignment from its Jukes-Cantor distances. */
    public static TreeInterface buildUpgmaTree(Alignment alignment) {
        ClusterTree tree = new ClusterTree();
        tree.initByName("clusterType", ClusterTree.Type.upgma, "taxa", alignment);
        return tree;
    }

    /** Returns the Fitch state set of the given node at the given pattern, as a bit mask over the states. */
    public long getStateSet(Node node, int patternNr) {
        return this.stateSets[node.getNr()][patternNr];
    }

    /** Returns the parsimony score of the given pattern. */
    public int getPatternScore(int patternNr) {
        return this.patternScores[patternNr];
    }

    /** Returns the parsimony score per site of the alignment. */
    public int[] getSiteScores() {
        int[] siteScores = new int[this.alignment.getSiteCount()];
        for (int site = 0; site < siteScores.length; site++) {
            siteScores[site] = this.patternScores[this.alignment.getPatternIndex(site)];
        }
        return siteScores;
    }

    /**
     * Computes the state sets of every node and the score of every pattern. A tip's set
     * contains all states compatible with its observed character; an inner node's set is
     * the intersection of its children's sets, or their union if they do not intersect,
     * which costs one change.
     */
    private void computeStateSets() {
        DataType dataType = this.alignment.getDataType();
        int numPatterns = this.alignment.getPatternCount();

        for (Node node : this.tree.listNodesPostOrder(null, null)) {
            long[] nodeStateSets = this.stateSets[node.getNr()];

            if (node.isLeaf()) {
                int taxonNr = this.alignment.getTaxonIndex(node.getID());
                if (taxonNr < 0) {
                    throw new IllegalArgumentException("Tip " + node.getID() + " is not in the alignment.");
                }

                for (int patternNr = 0; patternNr < numPatterns; patternNr++) {
                    nodeStateSets[patternNr] = getStateSet(dataType, this.alignment.getPattern(taxonNr, patternNr));
                }

                continue;
            }

            for (int patternNr = 0; patternNr < numPatterns; patternNr++) {
                long intersection = ~0L;
                long union = 0L;

                for (Node child : node.getChildren()) {
                    intersection &= this.stateSets[child.getNr()][patternNr];
                    union |= this.stateSets[child.getNr()][patternNr];
                }

                if (intersection != 0L) {
                    nodeStateSets[patternNr] = intersection;
                } else {
                    nodeStateSets[patternNr] = union;
                    this.patternScores[patternNr]++;
                }
            }
        }
    }

    /** Returns the set of states compatible with the given character code, as a bit mask. */
    private static long getStateSet(DataType dataType, int code) {
        long stateSet = 0L;
        for (int state : dataType.getStatesForCode(code)) {
            stateSet |= 1L << state;
        }
        return stateSet;
    }

}
