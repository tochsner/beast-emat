package emat.helper;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.datatype.DataType;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import emat.state.Mutations;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * The tips whose data are close to the data of every tip, i.e. differ from it at no more
 * than a given number of sites. Two tips differ at a site if no state is compatible with
 * both of their data. The distances only depend on the data, so they stay fixed while the
 * tree changes.
 * <p>
 * The data of every tip are kept as their differences to a base sequence, the most common
 * single state at every site, so that their cost follows the divergence of the tips rather
 * than the genome length. Tips with identical data form a class, which is close to itself
 * and has the same close classes for all its tips. Hence, the closeness is kept between
 * classes, which stays small however many tips share their data.
 */
public final class TipNeighbours {

    // per node: the class of its data, which is -1 for inner nodes
    private final int[] classOfNode;

    // per class: its tips, and the other classes whose data is close
    private final int[][] membersOfClass;
    private final int[][] neighbourClassesOfClass;

    // the base sequence, and the sorted sites where it differs from the reference sequence of the mutations
    private final int[] baseSequence;
    private final int[] baseChangeSites;

    // per class: the sorted sites where the data is a single state other than the base state, and these states
    private final int[][] definiteSitesOfClass;
    private final int[][] definiteStatesOfClass;

    public TipNeighbours(Mutations mutations, Tree tree, int maxDistance) {
        Alignment alignment = mutations.getAlignment();
        DataType dataType = alignment.getDataType();
        int numStates = alignment.getMaxStateCount();
        int[] referenceSequence = mutations.getReferenceSequence();
        List<Node> tips = tree.getExternalNodes();

        this.baseSequence = computeBaseSequence(alignment, tips, referenceSequence);
        this.baseChangeSites = IntStream.range(0, referenceSequence.length)
                .filter(site -> this.baseSequence[site] != referenceSequence[site])
                .toArray();

        // per tip, the sorted sites where its data is neither the base state nor missing, with the bit masks of the compatible states

        this.classOfNode = new int[tree.getNodeCount()];
        Arrays.fill(this.classOfNode, -1);

        Map<TipData, Integer> classNrs = new HashMap<>();
        List<TipData> classData = new ArrayList<>();
        List<List<Integer>> classMembers = new ArrayList<>();

        int[] siteBuffer = new int[referenceSequence.length];
        int[] maskBuffer = new int[referenceSequence.length];
        for (Node tip : tips) {
            int taxonIndex = alignment.getTaxonIndex(tip.getID());

            int numSites = 0;
            for (int site = 0; site < referenceSequence.length; site++) {
                int[] states = dataType.getStatesForCode(alignment.getPattern(taxonIndex, alignment.getPatternIndex(site)));
                int mask = 0;
                for (int state : states) {
                    mask |= 1 << state;
                }

                if (states.length < numStates && mask != 1 << this.baseSequence[site]) {
                    siteBuffer[numSites] = site;
                    maskBuffer[numSites] = mask;
                    numSites++;
                }
            }

            TipData data = new TipData(Arrays.copyOf(siteBuffer, numSites), Arrays.copyOf(maskBuffer, numSites));
            Integer classNr = classNrs.get(data);
            if (classNr == null) {
                classNr = classData.size();
                classNrs.put(data, classNr);
                classData.add(data);
                classMembers.add(new ArrayList<>());
            }
            classMembers.get(classNr).add(tip.getNr());
            this.classOfNode[tip.getNr()] = classNr;
        }

        int numClasses = classData.size();
        this.membersOfClass = new int[numClasses][];
        this.definiteSitesOfClass = new int[numClasses][];
        this.definiteStatesOfClass = new int[numClasses][];
        int[][] sitesOfClass = new int[numClasses][];
        int[][] masksOfClass = new int[numClasses][];
        for (int classNr = 0; classNr < numClasses; classNr++) {
            this.membersOfClass[classNr] = classMembers.get(classNr).stream().mapToInt(Integer::intValue).toArray();
            sitesOfClass[classNr] = classData.get(classNr).sites();
            masksOfClass[classNr] = classData.get(classNr).masks();

            int[] classSites = sitesOfClass[classNr];
            int[] classMasks = masksOfClass[classNr];
            int[] definiteSlots = IntStream.range(0, classSites.length)
                    .filter(slot -> Integer.bitCount(classMasks[slot]) == 1)
                    .toArray();
            this.definiteSitesOfClass[classNr] = Arrays.stream(definiteSlots).map(slot -> classSites[slot]).toArray();
            this.definiteStatesOfClass[classNr] = Arrays.stream(definiteSlots).map(slot -> Integer.numberOfTrailingZeros(classMasks[slot])).toArray();
        }

        this.neighbourClassesOfClass = findNeighbourClasses(sitesOfClass, masksOfClass, this.baseSequence, numStates, maxDistance);
    }

    /**
     * Returns the most common single state of the tips at every site, or the reference
     * state where no tip has a single state.
     */
    private static int[] computeBaseSequence(Alignment alignment, List<Node> tips, int[] referenceSequence) {
        DataType dataType = alignment.getDataType();
        int[][] stateCounts = new int[referenceSequence.length][alignment.getMaxStateCount()];
        for (Node tip : tips) {
            int taxonIndex = alignment.getTaxonIndex(tip.getID());
            for (int site = 0; site < referenceSequence.length; site++) {
                int[] states = dataType.getStatesForCode(alignment.getPattern(taxonIndex, alignment.getPatternIndex(site)));
                if (states.length == 1) {
                    stateCounts[site][states[0]]++;
                }
            }
        }

        int[] baseSequence = referenceSequence.clone();
        for (int site = 0; site < referenceSequence.length; site++) {
            for (int state = 0; state < stateCounts[site].length; state++) {
                if (stateCounts[site][state] > stateCounts[site][baseSequence[site]]) {
                    baseSequence[site] = state;
                }
            }
        }
        return baseSequence;
    }

    /**
     * Returns for every class the sorted numbers of the other classes within the given
     * distance. The classes are given by their sorted sites where the data is not the base
     * state, with the bit masks of the compatible states there, and are all different.
     * <p>
     * Up to a distance of one, only a few pairs of classes are close, so they are looked up
     * instead of comparing all pairs: a class without ambiguous data differs from another one
     * at a single site only if it has one entry more or fewer, or an entry with another state
     * at the same site, which makes both equal once that entry is removed. Classes with
     * ambiguous data can be close to classes that differ in many entries, so they are
     * compared with every class.
     */
    static int[][] findNeighbourClasses(int[][] sites, int[][] masks, int[] baseSequence, int numStates, int maxDistance) {
        NeighbourFinder finder = new NeighbourFinder(sites, masks, baseSequence, numStates, maxDistance);

        if (maxDistance > 1) {
            for (int classNr = 0; classNr < sites.length; classNr++) {
                for (int otherNr = classNr + 1; otherNr < sites.length; otherNr++) {
                    finder.addIfClose(classNr, otherNr);
                }
            }
        } else {
            finder.addAmbiguousClasses();
            if (maxDistance == 1) {
                finder.addCleanClasses();
            }
        }
        return finder.getNeighbourClasses();
    }

    /** Collects the pairs of close classes. */
    private static final class NeighbourFinder {

        private static final long HIGH_BITS = 0xFFFFFFFF00000000L;

        private final int[][] sites;
        private final int[][] masks;
        private final int[] baseSequence;
        private final int numStates;
        private final int maxDistance;
        private final List<List<Integer>> neighbours = new ArrayList<>();
        private final boolean[] isAmbiguous;

        NeighbourFinder(int[][] sites, int[][] masks, int[] baseSequence, int numStates, int maxDistance) {
            this.sites = sites;
            this.masks = masks;
            this.baseSequence = baseSequence;
            this.numStates = numStates;
            this.maxDistance = maxDistance;

            this.isAmbiguous = new boolean[sites.length];
            for (int classNr = 0; classNr < sites.length; classNr++) {
                this.neighbours.add(new ArrayList<>());
                this.isAmbiguous[classNr] = Arrays.stream(masks[classNr]).anyMatch(mask -> Integer.bitCount(mask) > 1);
            }
        }

        /** Compares every class with ambiguous data to all the other classes, every pair once. */
        void addAmbiguousClasses() {
            for (int classNr = 0; classNr < this.sites.length; classNr++) {
                if (!this.isAmbiguous[classNr]) {
                    continue;
                }
                for (int otherNr = 0; otherNr < this.sites.length; otherNr++) {
                    if (otherNr != classNr && (!this.isAmbiguous[otherNr] || classNr < otherNr)) {
                        this.addIfClose(classNr, otherNr);
                    }
                }
            }
        }

        /**
         * Finds the pairs of classes without ambiguous data that differ at one site. A set
         * of entries (site, state) gets the sum of a random-looking hash per entry, which
         * lets us hash the set without one entry in constant time. Two classes with another
         * state at one site have the same hash without that entry. The hash also has the
         * site mixed in, so that only classes that differ at the same site can meet. A class
         * with an entry more than another one is found by looking up its hash without that
         * entry. Equal hashes by chance only cost a comparison, as the pairs are checked.
         */
        void addCleanClasses() {
            int numClasses = this.sites.length;
            long[] hashes = new long[numClasses];
            Map<Long, List<Integer>> classesOfHash = new HashMap<>();
            int numEntries = 0;
            for (int classNr = 0; classNr < numClasses; classNr++) {
                if (this.isAmbiguous[classNr]) {
                    continue;
                }
                for (int slot = 0; slot < this.sites[classNr].length; slot++) {
                    hashes[classNr] += this.computeEntryHash(classNr, slot);
                }
                classesOfHash.computeIfAbsent(hashes[classNr], hash -> new ArrayList<>()).add(classNr);
                numEntries += this.sites[classNr].length;
            }

            // the keys keep the upper half of the hash without an entry and the site, and the class in the lower half
            long[] keys = new long[numEntries];
            int numKeys = 0;
            for (int classNr = 0; classNr < numClasses; classNr++) {
                if (this.isAmbiguous[classNr]) {
                    continue;
                }
                for (int slot = 0; slot < this.sites[classNr].length; slot++) {
                    long hashWithoutEntry = hashes[classNr] - this.computeEntryHash(classNr, slot);
                    for (int otherNr : classesOfHash.getOrDefault(hashWithoutEntry, List.of())) {
                        this.addIfClose(classNr, otherNr);
                    }
                    keys[numKeys++] = ((hashWithoutEntry + mix(~this.sites[classNr][slot])) & HIGH_BITS) | classNr;
                }
            }

            Arrays.sort(keys);
            int start = 0;
            while (start < keys.length) {
                int end = start + 1;
                while (end < keys.length && (keys[end] & HIGH_BITS) == (keys[start] & HIGH_BITS)) {
                    end++;
                }
                for (int i = start; i < end; i++) {
                    for (int j = i + 1; j < end; j++) {
                        this.addIfClose((int) keys[i], (int) keys[j]);
                    }
                }
                start = end;
            }
        }

        private long computeEntryHash(int classNr, int slot) {
            int state = Integer.numberOfTrailingZeros(this.masks[classNr][slot]);
            return mix((long) this.sites[classNr][slot] * this.numStates + state);
        }

        /** Mixes the bits of the given number, so that the hashes of different entries look random. */
        private static long mix(long value) {
            long z = value + 0x9E3779B97F4A7C15L;
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            return z ^ (z >>> 31);
        }

        /** Adds the pair to the neighbours of both classes if their distance is small enough. */
        void addIfClose(int classNr, int otherNr) {
            if (classNr == otherNr) {
                return;
            }

            int distance = computeDistance(this.sites[classNr], this.masks[classNr], this.sites[otherNr], this.masks[otherNr], this.baseSequence, this.maxDistance);
            if (distance <= this.maxDistance) {
                this.neighbours.get(classNr).add(otherNr);
                this.neighbours.get(otherNr).add(classNr);
            }
        }

        int[][] getNeighbourClasses() {
            return this.neighbours.stream()
                    .map(list -> list.stream().mapToInt(Integer::intValue).sorted().distinct().toArray())
                    .toArray(int[][]::new);
        }
    }

    /**
     * Counts the sites where the given classes differ by merging their sorted sites, where a
     * site of only one class has the base state at the other. Stops once the count
     * exceeds the given maximum.
     */
    private static int computeDistance(int[] sites, int[] masks, int[] otherSites, int[] otherMasks, int[] baseSequence, int maxDistance) {
        int distance = 0;
        int i = 0;
        int j = 0;
        while ((i < sites.length || j < otherSites.length) && distance <= maxDistance) {
            int site = i < sites.length ? sites[i] : Integer.MAX_VALUE;
            int otherSite = j < otherSites.length ? otherSites[j] : Integer.MAX_VALUE;

            int mask;
            int otherMask;
            if (site == otherSite) {
                mask = masks[i++];
                otherMask = otherMasks[j++];
            } else if (site < otherSite) {
                mask = masks[i++];
                otherMask = 1 << baseSequence[site];
            } else {
                mask = 1 << baseSequence[otherSite];
                otherMask = otherMasks[j++];
            }

            if ((mask & otherMask) == 0) {
                distance++;
            }
        }
        return distance;
    }

    /** The sorted sites where the data of a tip is not the base state, with the bit masks of the compatible states. */
    private record TipData(int[] sites, int[] masks) {

        @Override
        public boolean equals(Object other) {
            return other instanceof TipData data && Arrays.equals(this.sites, data.sites) && Arrays.equals(this.masks, data.masks);
        }

        @Override
        public int hashCode() {
            return 31 * Arrays.hashCode(this.sites) + Arrays.hashCode(this.masks);
        }
    }

    /** Returns the number of classes of tips with identical data. */
    public int getNumClasses() {
        return this.membersOfClass.length;
    }

    /** Returns the class of the given tip. */
    public int getClassNr(Node tip) {
        return this.classOfNode[tip.getNr()];
    }

    /** Returns the numbers of the tips in the given class. */
    public int[] getMembers(int classNr) {
        return this.membersOfClass[classNr];
    }

    /** Returns the numbers of the other classes that are close to the given class. */
    public int[] getNeighbourClasses(int classNr) {
        return this.neighbourClassesOfClass[classNr];
    }

    /** Returns the most common single state of the tips at every site. */
    public int[] getBaseSequence() {
        return this.baseSequence;
    }

    /** Returns the sorted sites where the base sequence differs from the reference sequence of the mutations. */
    public int[] getBaseChangeSites() {
        return this.baseChangeSites;
    }

    /** Returns the sorted sites where the data of the given tip is a single state other than the base state. */
    public int[] getDefiniteSites(Node tip) {
        return this.definiteSitesOfClass[this.getClassNr(tip)];
    }

    /** Returns the states of the given tip at its definite sites, in the order of getDefiniteSites. */
    public int[] getDefiniteStates(Node tip) {
        return this.definiteStatesOfClass[this.getClassNr(tip)];
    }

}
