package emat.alignment;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.alignment.Alignment;
import beast.base.spec.evolution.alignment.FilteredAlignment;

import java.util.stream.IntStream;

@Description("Alignment that keeps the sites of another alignment at which some taxon is observed, or the sites at " +
        "which every taxon is missing. A taxon is missing at a site if its character, such as N, ? or a gap, is " +
        "compatible with every state. A site at which every taxon is missing does not change the likelihood of any " +
        "tree, so dropping it leaves the posterior of the tree and the parameters unchanged, while its history would " +
        "only follow the prior. The same alignment with opposite flags splits the sites into two disjoint alignments " +
        "that together contain every site. The alignment has no sites if there are none of the requested kind.")
public class MissingSiteFilteredAlignment extends FilteredAlignment {

    final public Input<Boolean> missingInput = new Input<>("missing", "if true, keep the sites at which every taxon is missing, otherwise keep all other sites", false);

    public MissingSiteFilteredAlignment() {
        this.filterInput.setRule(Input.Validate.OPTIONAL);
    }

    @Override
    public void initAndValidate() {
        if (this.filterInput.get() != null) {
            throw new IllegalArgumentException("The filter of a MissingSiteFilteredAlignment is derived from the sites and must not be set.");
        }

        Alignment alignment = this.alignmentInput.get();
        boolean missing = this.missingInput.get();

        boolean[] isMissingPattern = this.computeMissingPatterns(alignment);
        int[] selectedSites = IntStream.range(0, alignment.getSiteCount())
                .filter(site -> isMissingPattern[alignment.getPatternIndex(site)] == missing)
                .toArray();

        this.filterInput.setValue(this.getFilterSpec(selectedSites), this);
        super.initAndValidate();
    }

    /** Returns for every pattern whether every taxon has a character that is compatible with every state. */
    private boolean[] computeMissingPatterns(Alignment alignment) {
        boolean[] isMissingPattern = new boolean[alignment.getPatternCount()];
        int numStates = alignment.getDataType().getStateCount();

        for (int patternNr = 0; patternNr < isMissingPattern.length; patternNr++) {
            boolean isMissingForAllTaxa = true;
            for (int taxonNr = 0; taxonNr < alignment.getTaxonCount() && isMissingForAllTaxa; taxonNr++) {
                int code = alignment.getPattern(taxonNr, patternNr);
                isMissingForAllTaxa = alignment.getDataType().getStatesForCode(code).length == numStates;
            }

            isMissingPattern[patternNr] = isMissingForAllTaxa;
        }

        return isMissingPattern;
    }

    /** Returns the filter specification of the given sorted sites, with consecutive sites merged into ranges. */
    private String getFilterSpec(int[] sortedSites) {
        // the filter needs a non-empty specification, and the empty range 2-1 selects no sites
        if (sortedSites.length == 0) {
            return "2-1";
        }

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
