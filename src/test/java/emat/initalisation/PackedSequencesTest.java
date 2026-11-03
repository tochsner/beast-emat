package emat.initalisation;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class PackedSequencesTest {

    /** Checks the counts of differences against a direct comparison of sequences that span several words. */
    @Test
    public void testDifferencesMatchDirectCount() {
        String[] sequences = this.createRandomSequences(6, 300, new Random(1));
        Alignment alignment = this.createAlignment(sequences);
        PackedSequences packed = new PackedSequences(alignment, this.createNames(sequences.length));

        for (int tip1 = 0; tip1 < sequences.length; tip1++) {
            for (int tip2 = 0; tip2 < sequences.length; tip2++) {
                int expected = this.countDifferences(sequences[tip1], sequences[tip2]);
                assertEquals(expected, packed.countDifferences(tip1, tip2, Integer.MAX_VALUE));
            }
        }
    }

    /** Checks that a count that reaches the bound returns a number of at least the bound. */
    @Test
    public void testDifferencesStopAtBound() {
        String[] sequences = this.createRandomSequences(2, 300, new Random(2));
        PackedSequences packed = new PackedSequences(this.createAlignment(sequences), this.createNames(2));

        int numDifferences = this.countDifferences(sequences[0], sequences[1]);
        assertTrue(packed.countDifferences(0, 1, 10) >= 10);
        assertEquals(numDifferences, packed.countDifferences(0, 1, numDifferences + 1));
    }

    /** Checks that invariant sites and sites where only one tip has a valid state are left out. */
    @Test
    public void testOnlyVariableSitesArePacked() {
        String[] sequences = {"AACNA", "AAGNC", "AAGTC"};
        PackedSequences packed = new PackedSequences(this.createAlignment(sequences), this.createNames(3));

        assertEquals(5, packed.getNumSites());
        assertEquals(2, packed.getNumVariableSites());
    }

    /**
     * Checks the Fitch parsimony changes on ((A,B),(C,D)) at sites that need one, two and three
     * changes, and at a site where a missing state is resolved without a change.
     */
    @Test
    public void testParsimonyChanges() {
        String[] sequences = {
                "AAAAA",
                "AACCN",
                "ACAGC",
                "ACCTC"
        };
        PackedSequences packed = new PackedSequences(this.createAlignment(sequences), this.createNames(4));
        BinaryTopology topology = new BinaryTopology(4, new int[]{4, 4, 5, 5, 6, 6, -1});

        int[] numChanges = packed.countParsimonyChanges(topology);

        int total = 0;
        for (int changes : numChanges) {
            total += changes;
        }
        assertEquals(7, total);
        assertEquals(0, numChanges[topology.getRoot()]);
    }

    private String[] createRandomSequences(int numSequences, int length, Random random) {
        String characters = "ACGTN";
        String[] sequences = new String[numSequences];
        for (int i = 0; i < numSequences; i++) {
            StringBuilder sequence = new StringBuilder();
            for (int site = 0; site < length; site++) {
                sequence.append(characters.charAt(random.nextInt(characters.length())));
            }
            sequences[i] = sequence.toString();
        }
        return sequences;
    }

    private Alignment createAlignment(String[] sequences) {
        List<Sequence> sequenceList = new ArrayList<>();
        for (int i = 0; i < sequences.length; i++) {
            sequenceList.add(new Sequence("t" + i, sequences[i]));
        }

        Alignment alignment = new Alignment();
        alignment.initByName("sequence", sequenceList, "dataType", "nucleotide");
        return alignment;
    }

    private String[] createNames(int numSequences) {
        String[] names = new String[numSequences];
        for (int i = 0; i < numSequences; i++) {
            names[i] = "t" + i;
        }
        return names;
    }

    private int countDifferences(String sequence1, String sequence2) {
        int numDifferences = 0;
        for (int site = 0; site < sequence1.length(); site++) {
            char state1 = sequence1.charAt(site);
            char state2 = sequence2.charAt(site);
            if (state1 != 'N' && state2 != 'N' && state1 != state2) {
                numDifferences++;
            }
        }
        return numDifferences;
    }

}
