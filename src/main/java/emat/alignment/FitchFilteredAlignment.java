package emat.alignment;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.datatype.DataType;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.spec.evolution.alignment.FilteredAlignment;
import beast.base.spec.evolution.tree.ClusterTree;

import java.util.stream.IntStream;

@Description("Alignment that keeps the sites of another alignment whose Fitch parsimony score on the UPGMA " +
        "tree of that alignment is at least, or below, a minimum score. The same minimum score with opposite " +
        "flags splits the sites into two disjoint alignments that together contain every site.")
public class FitchFilteredAlignment extends FilteredAlignment {

    final public Input<Integer> minScoreInput = new Input<>("minScore", "the lowest parsimony score of a site in the high-score part", Input.Validate.REQUIRED);
    final public Input<Boolean> highestInput = new Input<>("highest", "if true, keep the sites with a parsimony score of at least minScore, otherwise keep all other sites", true);

    // the Fitch parsimony score per site of the input alignment
    int[] siteScores;

    public FitchFilteredAlignment() {
        this.filterInput.setRule(Input.Validate.OPTIONAL);
    }

    @Override
    public void initAndValidate() {
        if (this.filterInput.get() != null) {
            throw new IllegalArgumentException("The filter of a FitchFilteredAlignment is derived from the parsimony scores and must not be set.");
        }

        int minScore = this.minScoreInput.get();
        boolean highest = this.highestInput.get();

        this.siteScores = this.computeSiteScores();
        int[] selectedSites = IntStream.range(0, this.siteScores.length)
                .filter(site -> (this.siteScores[site] >= minScore) == highest)
                .toArray();

        if (selectedSites.length == 0) {
            throw new IllegalArgumentException("No sites are selected with minScore " + minScore + " and highest=" + highest + ".");
        }

        this.filterInput.setValue(this.getFilterSpec(selectedSites), this);
        super.initAndValidate();
    }

    /** Returns the Fitch parsimony score per site of the input alignment. */
    public int[] getSiteScores() {
        return this.siteScores.clone();
    }

    /* Site Selection */

    /** Returns the filter specification of the given sorted sites, with consecutive sites merged into ranges. */
    private String getFilterSpec(int[] sortedSites) {
        StringBuilder filterSpec = new StringBuilder();

        int rangeStart = 0;
        while (rangeStart < sortedSites.length) {
            int rangeEnd = rangeStart;
            while (rangeEnd + 1 < sortedSites.length && sortedSites[rangeEnd + 1] == sortedSites[rangeEnd] + 1) {
                rangeEnd++;
            }

            if (filterSpec.length() > 0) {
                filterSpec.append(',');
            }

            // filter specifications count sites from one
            filterSpec.append(sortedSites[rangeStart] + 1).append('-').append(sortedSites[rangeEnd] + 1);

            rangeStart = rangeEnd + 1;
        }

        return filterSpec.toString();
    }

    /* Fitch Parsimony */

    /**
     * Computes the Fitch parsimony score per site of the input alignment on its UPGMA tree.
     * The scores are computed per pattern and then assigned to every site of the pattern.
     */
    private int[] computeSiteScores() {
        Alignment alignment = this.alignmentInput.get();
        TreeInterface tree = this.buildUpgmaTree(alignment);
        DataType dataType = alignment.getDataType();

        if (alignment.getMaxStateCount() > Long.SIZE) {
            throw new IllegalArgumentException("Fitch filtering supports at most " + Long.SIZE + " states.");
        }

        int numPatterns = alignment.getPatternCount();
        long[][] stateSets = new long[tree.getNodeCount()][numPatterns];
        int[] patternScores = new int[numPatterns];

        // a node's state set is the intersection of its children's sets, or their union if they do not intersect

        for (Node node : tree.listNodesPostOrder(null, null)) {
            long[] nodeStateSets = stateSets[node.getNr()];

            if (node.isLeaf()) {
                int taxonNr = alignment.getTaxonIndex(node.getID());
                if (taxonNr < 0) {
                    throw new IllegalArgumentException("Tip " + node.getID() + " is not in the alignment.");
                }

                for (int patternNr = 0; patternNr < numPatterns; patternNr++) {
                    nodeStateSets[patternNr] = this.getStateSet(dataType, alignment.getPattern(taxonNr, patternNr));
                }

                continue;
            }

            for (int patternNr = 0; patternNr < numPatterns; patternNr++) {
                long intersection = ~0L;
                long union = 0L;

                for (Node child : node.getChildren()) {
                    intersection &= stateSets[child.getNr()][patternNr];
                    union |= stateSets[child.getNr()][patternNr];
                }

                if (intersection != 0L) {
                    nodeStateSets[patternNr] = intersection;
                } else {
                    nodeStateSets[patternNr] = union;
                    patternScores[patternNr]++;
                }
            }
        }

        int[] siteScores = new int[alignment.getSiteCount()];
        for (int site = 0; site < siteScores.length; site++) {
            siteScores[site] = patternScores[alignment.getPatternIndex(site)];
        }

        return siteScores;
    }

    /** Builds the UPGMA tree of the given alignment from its Jukes-Cantor distances. */
    private TreeInterface buildUpgmaTree(Alignment alignment) {
        ClusterTree tree = new ClusterTree();
        tree.initByName("clusterType", ClusterTree.Type.upgma, "taxa", alignment);
        return tree;
    }

    /** Returns the set of states compatible with the given character code, as a bit mask. */
    private long getStateSet(DataType dataType, int code) {
        long stateSet = 0L;
        for (int state : dataType.getStatesForCode(code)) {
            stateSet |= 1L << state;
        }
        return stateSet;
    }

}
