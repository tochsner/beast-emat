package emat;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import emat.alignment.MissingSiteFilteredAlignment;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class MissingSiteFilteredAlignmentTest {

    /** Checks that only sites at which every taxon is missing count as missing, and that both flags split the sites. */
    @Test
    public void testPartsAreComplementary() {
        Alignment alignment = new Alignment();
        alignment.initByName(
                "sequence", List.of(
                        new Sequence("A", "AN-?RC"),
                        new Sequence("B", "NN-NNC"),
                        new Sequence("C", "NN--NC")
                ),
                "dataType", "nucleotide"
        );

        // sites: 0 one taxon observed, 1 all N, 2 all gaps, 3 mixed missing characters, 4 one ambiguity code, 5 constant
        assertArrayEquals(new int[]{0, 4, 5}, this.createFilteredAlignment(alignment, false).indices());
        assertArrayEquals(new int[]{1, 2, 3}, this.createFilteredAlignment(alignment, true).indices());
    }

    /** Checks that a selection without any site gives an alignment without sites. */
    @Test
    public void testEmptySelectionHasNoSites() {
        Alignment alignment = new Alignment();
        alignment.initByName(
                "sequence", List.of(new Sequence("A", "AC"), new Sequence("B", "NC")),
                "dataType", "nucleotide"
        );

        MissingSiteFilteredAlignment missingSites = this.createFilteredAlignment(alignment, true);
        assertEquals(0, missingSites.getSiteCount());
        assertEquals(0, missingSites.getPatternCount());
        assertEquals(2, this.createFilteredAlignment(alignment, false).getSiteCount());
    }

    private MissingSiteFilteredAlignment createFilteredAlignment(Alignment alignment, boolean missing) {
        MissingSiteFilteredAlignment filteredAlignment = new MissingSiteFilteredAlignment();
        filteredAlignment.initByName("data", alignment, "missing", missing);
        return filteredAlignment;
    }

}
