package emat;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import emat.alignment.FitchFilteredAlignment;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class FitchFilteredAlignmentTest {

    /**
     * Checks the parsimony scores of sites that need zero to three changes on the UPGMA
     * tree ((A,B),(C,D)), which the last three sites enforce.
     */
    @Test
    public void testSiteScores() {
        FitchFilteredAlignment alignment = this.createFilteredAlignment(2, true);

        assertArrayEquals(new int[]{0, 1, 2, 3, 1, 1, 1, 1}, alignment.getSiteScores());
    }

    /** Checks that both flags with the same minimum score split the sites into the high- and low-score parts. */
    @Test
    public void testPartsAreComplementary() {
        FitchFilteredAlignment highAlignment = this.createFilteredAlignment(2, true);
        FitchFilteredAlignment lowAlignment = this.createFilteredAlignment(2, false);

        assertArrayEquals(new int[]{2, 3}, highAlignment.indices());
        assertArrayEquals(new int[]{0, 1, 4, 5, 6, 7}, lowAlignment.indices());

        assertEquals(2, highAlignment.getSiteCount());
        assertEquals(6, lowAlignment.getSiteCount());
    }

    /** Checks that a selection without any site is rejected. */
    @Test
    public void testEmptySelectionIsRejected() {
        assertThrows(RuntimeException.class, () -> this.createFilteredAlignment(4, true));
        assertThrows(RuntimeException.class, () -> this.createFilteredAlignment(0, false));
    }

    private FitchFilteredAlignment createFilteredAlignment(int minScore, boolean highest) {
        Alignment alignment = new Alignment();
        alignment.initByName(
                "sequence", List.of(
                        new Sequence("A", "AAAAAAAC"),
                        new Sequence("B", "AAGCAAAC"),
                        new Sequence("C", "AGAGAGGT"),
                        new Sequence("D", "AGGTGGGT")
                ),
                "dataType", "nucleotide"
        );

        FitchFilteredAlignment filteredAlignment = new FitchFilteredAlignment();
        filteredAlignment.initByName(
                "data", alignment,
                "minScore", minScore,
                "highest", highest
        );

        return filteredAlignment;
    }

}
