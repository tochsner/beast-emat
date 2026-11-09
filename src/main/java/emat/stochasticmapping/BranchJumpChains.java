package emat.stochasticmapping;

import beast.base.util.Randomizer;
import emat.state.Mutation;

import java.util.Arrays;
import java.util.List;

/**
 * Collects the jump chains of the sites on one branch and turns them into mutations sorted by
 * descending height without sorting. Given the number of jumps of every chain, the jump times
 * of all chains are independent and uniform on the branch, and independent of the states the
 * chains visit. The times of all jumps in increasing order are thus uniform order statistics,
 * and the order in which the chains jump is a uniformly random arrangement of their jumps,
 * independent of the times. Both are drawn in time linear in the number of jumps, and the k-th
 * jump of a chain then takes the k-th of its times. A chain may contain jumps that stay in the
 * same state, as uniformisation does, which are dropped after they took their times.
 */
class BranchJumpChains {

    private int numChains = 0;
    private int numJumps = 0;

    // per chain: its site, its states (the start state followed by the state after every jump),
    // the number of its jumps already placed and the height of its last mutation
    private int[] sites = new int[16];
    private int[][] chains = new int[16][];
    private int[] placedJumps = new int[16];
    private double[] previousHeights = new double[16];

    // per jump in time order: the chain it belongs to and its time since the branch start
    private int[] jumpChains = new int[64];
    private double[] jumpTimes = new double[64];

    /**
     * Adds the jump chain of the given site, which holds the start state followed by the state
     * after every jump. The array is read when the mutations are added.
     */
    void addChain(int site, int[] states) {
        if (states.length < 2) {
            return;
        }

        if (this.numChains == this.sites.length) {
            int capacity = 2 * this.sites.length;
            this.sites = Arrays.copyOf(this.sites, capacity);
            this.chains = Arrays.copyOf(this.chains, capacity);
            this.placedJumps = Arrays.copyOf(this.placedJumps, capacity);
            this.previousHeights = Arrays.copyOf(this.previousHeights, capacity);
        }

        this.sites[this.numChains] = site;
        this.chains[this.numChains] = states;
        this.numChains++;
        this.numJumps += states.length - 1;
    }

    /**
     * Places the jumps of all collected chains on the branch above the given node, which starts
     * at the given height and lasts the given duration, adds the jumps that change the state as
     * mutations sorted by descending height, and clears the chains.
     */
    void addMutations(List<Mutation> branchMutations, int nodeNr, double branchStartHeight, double duration) {
        if (this.jumpChains.length < this.numJumps) {
            int capacity = Math.max(this.numJumps, 2 * this.jumpChains.length);
            this.jumpChains = new int[capacity];
            this.jumpTimes = new double[capacity];
        }

        this.arrangeJumps();
        this.sampleSortedTimes(duration);

        Arrays.fill(this.placedJumps, 0, this.numChains, 0);
        Arrays.fill(this.previousHeights, 0, this.numChains, Mutation.BRANCH_START);

        for (int jump = 0; jump < this.numJumps; jump++) {
            int chain = this.jumpChains[jump];
            int[] states = this.chains[chain];
            int position = this.placedJumps[chain]++;

            if (states[position] != states[position + 1]) {
                double height = branchStartHeight - this.jumpTimes[jump];
                branchMutations.add(new Mutation(nodeNr, height, this.previousHeights[chain], this.sites[chain],
                        states[position], states[position + 1]));
                this.previousHeights[chain] = height;
            }
        }

        Arrays.fill(this.chains, 0, this.numChains, null);
        this.numChains = 0;
        this.numJumps = 0;
    }

    /** Lists every chain once per jump and shuffles the list uniformly (Fisher-Yates). */
    private void arrangeJumps() {
        int jump = 0;
        for (int chain = 0; chain < this.numChains; chain++) {
            for (int i = 1; i < this.chains[chain].length; i++) {
                this.jumpChains[jump++] = chain;
            }
        }

        for (int i = this.numJumps - 1; i > 0; i--) {
            int j = Randomizer.nextInt(i + 1);
            int chain = this.jumpChains[i];
            this.jumpChains[i] = this.jumpChains[j];
            this.jumpChains[j] = chain;
        }
    }

    /**
     * Draws the uniform order statistics of all jumps on the given duration as the normalised
     * cumulative sums of standard exponential spacings, one more spacing than there are jumps.
     */
    private void sampleSortedTimes(double duration) {
        double cumulativeSpacing = 0.0;
        for (int jump = 0; jump < this.numJumps; jump++) {
            cumulativeSpacing -= Math.log1p(-Randomizer.nextDouble());
            this.jumpTimes[jump] = cumulativeSpacing;
        }
        cumulativeSpacing -= Math.log1p(-Randomizer.nextDouble());

        double scale = duration / cumulativeSpacing;
        for (int jump = 0; jump < this.numJumps; jump++) {
            this.jumpTimes[jump] *= scale;
        }
    }

}
