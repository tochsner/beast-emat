package emat.alignment;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.alignment.Alignment;
import beast.base.spec.evolution.alignment.FilteredAlignment;

import java.util.stream.IntStream;

@Description("Alignment that keeps the sites of another alignment that are fully constant, or all other sites. A " +
        "site is fully constant if every taxon has the same unambiguous state there, so no taxon has a gap or " +
        "missing data. The same alignment with opposite flags splits the sites into two disjoint alignments that " +
        "together contain every site.")
public class ConstantSiteFilteredAlignment extends FilteredAlignment {

    final public Input<Boolean> constantInput = new Input<>("constant", "if true, keep the fully constant sites, otherwise keep all other sites", true);

    public ConstantSiteFilteredAlignment() {
        this.filterInput.setRule(Input.Validate.OPTIONAL);
    }

    @Override
    public void initAndValidate() {
        if (this.filterInput.get() != null) {
            throw new IllegalArgumentException("The filter of a ConstantSiteFilteredAlignment is derived from the sites and must not be set.");
        }

        Alignment alignment = this.alignmentInput.get();
        boolean constant = this.constantInput.get();

        boolean[] isConstantPattern = this.computeConstantPatterns(alignment);
        int[] selectedSites = IntStream.range(0, alignment.getSiteCount())
                .filter(site -> isConstantPattern[alignment.getPatternIndex(site)] == constant)
                .toArray();

        if (selectedSites.length == 0) {
            throw new IllegalArgumentException("No sites are selected with constant=" + constant + ".");
        }

        this.filterInput.setValue(this.getFilterSpec(selectedSites), this);
        super.initAndValidate();
    }

    /** Returns for every pattern whether all taxa have the same unambiguous state in it. */
    private boolean[] computeConstantPatterns(Alignment alignment) {
        boolean[] isConstantPattern = new boolean[alignment.getPatternCount()];

        for (int patternNr = 0; patternNr < isConstantPattern.length; patternNr++) {
            int code = alignment.getPattern(0, patternNr);
            boolean isUnambiguous = alignment.getDataType().getStatesForCode(code).length == 1;

            boolean isSameForAllTaxa = true;
            for (int taxonNr = 1; taxonNr < alignment.getTaxonCount() && isSameForAllTaxa; taxonNr++) {
                isSameForAllTaxa = alignment.getPattern(taxonNr, patternNr) == code;
            }

            isConstantPattern[patternNr] = isUnambiguous && isSameForAllTaxa;
        }

        return isConstantPattern;
    }

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
