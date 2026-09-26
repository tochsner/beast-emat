package emat.alignment;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.spec.evolution.alignment.FilteredAlignment;
import emat.helper.FitchParsimony;

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

        this.siteScores = FitchParsimony.computeOnUpgmaTree(this.alignmentInput.get()).getSiteScores();
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

}
