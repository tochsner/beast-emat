package emat.operators.topology.mdspr;

import beast.base.util.Randomizer;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class TipNeighboursTest {

    private static final int NUM_STATES = 4;
    private static final int NUM_SITES = 12;

    /**
     * The neighbour classes found through hashing for distances of zero and one have to match
     * a comparison of all pairs, for classes with and without ambiguous data.
     */
    @Test
    public void testNeighbourClassesMatchAllPairs() {
        Randomizer.setSeed(3);
        int[] baseSequence = new int[NUM_SITES];
        for (int site = 0; site < NUM_SITES; site++) {
            baseSequence[site] = Randomizer.nextInt(NUM_STATES);
        }

        for (double ambiguityProbability : new double[]{0.0, 0.05, 0.3}) {
            int[][][] classes = this.createClasses(baseSequence, 150, ambiguityProbability);
            for (int maxDistance = 0; maxDistance <= 3; maxDistance++) {
                int[][] expected = this.compareAllPairs(classes, baseSequence, maxDistance);
                int[][] actual = TipNeighbours.findNeighbourClasses(classes[0], classes[1], baseSequence, NUM_STATES, maxDistance);
                assertArrayEquals(expected, actual, "ambiguity " + ambiguityProbability + ", distance " + maxDistance);
            }
        }
    }

    /**
     * Draws different classes of data that differ from the base sequence at a few sites, each
     * with the mask of its compatible states. A class returns its sites as the first and its
     * masks as the second entry of the result, where the result holds all classes in turn.
     */
    private int[][][] createClasses(int[] baseSequence, int numClasses, double ambiguityProbability) {
        Set<String> seen = new LinkedHashSet<>();
        int[][] sites = new int[numClasses][];
        int[][] masks = new int[numClasses][];
        int count = 0;
        while (count < numClasses) {
            int[] siteBuffer = new int[NUM_SITES];
            int[] maskBuffer = new int[NUM_SITES];
            int numSites = 0;
            for (int site = 0; site < NUM_SITES; site++) {
                if (Randomizer.nextDouble() > 0.2) {
                    continue;
                }

                int mask = 1 << Randomizer.nextInt(NUM_STATES);
                if (Randomizer.nextDouble() < ambiguityProbability) {
                    mask |= 1 << Randomizer.nextInt(NUM_STATES);
                }
                if (mask != 1 << baseSequence[site] && mask != (1 << NUM_STATES) - 1) {
                    siteBuffer[numSites] = site;
                    maskBuffer[numSites] = mask;
                    numSites++;
                }
            }

            int[] classSites = Arrays.copyOf(siteBuffer, numSites);
            int[] classMasks = Arrays.copyOf(maskBuffer, numSites);
            if (seen.add(Arrays.toString(classSites) + Arrays.toString(classMasks))) {
                sites[count] = classSites;
                masks[count] = classMasks;
                count++;
            }
        }
        return new int[][][]{sites, masks};
    }

    /** Computes the sorted neighbour classes by comparing the full sequences of every pair of classes. */
    private int[][] compareAllPairs(int[][][] classes, int[] baseSequence, int maxDistance) {
        int numClasses = classes[0].length;
        int[][] neighbours = new int[numClasses][];
        for (int classNr = 0; classNr < numClasses; classNr++) {
            int[] expanded = this.expand(classes, classNr, baseSequence);
            neighbours[classNr] = new int[0];
            for (int otherNr = 0; otherNr < numClasses; otherNr++) {
                int[] otherExpanded = this.expand(classes, otherNr, baseSequence);

                int distance = 0;
                for (int site = 0; site < NUM_SITES; site++) {
                    distance += (expanded[site] & otherExpanded[site]) == 0 ? 1 : 0;
                }
                if (otherNr != classNr && distance <= maxDistance) {
                    neighbours[classNr] = Arrays.copyOf(neighbours[classNr], neighbours[classNr].length + 1);
                    neighbours[classNr][neighbours[classNr].length - 1] = otherNr;
                }
            }
        }
        return neighbours;
    }

    /** Returns the mask of the compatible states of the class at every site. */
    private int[] expand(int[][][] classes, int classNr, int[] baseSequence) {
        int[] expanded = new int[NUM_SITES];
        for (int site = 0; site < NUM_SITES; site++) {
            expanded[site] = 1 << baseSequence[site];
        }
        for (int slot = 0; slot < classes[0][classNr].length; slot++) {
            expanded[classes[0][classNr][slot]] = classes[1][classNr][slot];
        }
        return expanded;
    }

}
