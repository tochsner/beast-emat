package emat;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import emat.alignment.ConstantSiteFilteredAlignment;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class ConstantSiteFilteredAlignmentTest {

    /** Checks that only sites without gaps, ambiguity or variation count as constant, and that both flags split the sites. */
    @Test
    public void testPartsAreComplementary() {
        Alignment alignment = this.createAlignment();

        // sites: 0 constant, 1 variable, 2 constant, 3 gap, 4 N, 5 constant, 6 ambiguity code, 7 variable with gap
        assertArrayEquals(new int[]{0, 2, 5}, this.createFilteredAlignment(alignment, true).indices());
        assertArrayEquals(new int[]{1, 3, 4, 6, 7}, this.createFilteredAlignment(alignment, false).indices());
    }

    /** Checks that a selection without any site gives an alignment without sites. */
    @Test
    public void testEmptySelectionHasNoSites() {
        Alignment alignment = new Alignment();
        alignment.initByName(
                "sequence", List.of(new Sequence("A", "AC"), new Sequence("B", "AC")),
                "dataType", "nucleotide"
        );

        ConstantSiteFilteredAlignment variableSites = this.createFilteredAlignment(alignment, false);
        assertEquals(0, variableSites.getSiteCount());
        assertEquals(0, variableSites.getPatternCount());
        assertEquals(2, this.createFilteredAlignment(alignment, true).getSiteCount());
    }

    private Alignment createAlignment() {
        Alignment alignment = new Alignment();
        alignment.initByName(
                "sequence", List.of(
                        new Sequence("A", "AAC-NTRA"),
                        new Sequence("B", "AGC-NTAA"),
                        new Sequence("C", "ACCCNTAC"),
                        new Sequence("D", "AACGNTAC")
                ),
                "dataType", "nucleotide"
        );
        return alignment;
    }

    private ConstantSiteFilteredAlignment createFilteredAlignment(Alignment alignment, boolean constant) {
        ConstantSiteFilteredAlignment filteredAlignment = new ConstantSiteFilteredAlignment();
        filteredAlignment.initByName("data", alignment, "constant", constant);
        return filteredAlignment;
    }

}
