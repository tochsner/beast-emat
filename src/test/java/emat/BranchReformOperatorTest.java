package emat;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeParser;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.evolution.branchratemodel.StrictClockModel;
import beast.base.spec.evolution.sitemodel.SiteModel;
import beast.base.spec.evolution.substitutionmodel.JukesCantor;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.util.Randomizer;
import emat.helper.MutationPaths;
import emat.operators.BranchReformOperator;
import emat.prior.GeneticPrior;
import emat.state.Mutation;
import emat.state.Mutations;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class BranchReformOperatorTest {

    private static final int NUM_BURNIN = 1000;
    private static final int NUM_SAMPLES = 200000;

    // nucleotide states
    private static final int A = 0;
    private static final int C = 1;
    private static final int G = 2;
    private static final int T = 3;

    /**
     * On the tree (A,B)R with tips A and C at a single site, every proposal resamples the
     * path A–R–B and thereby the root state. Under Jukes-Cantor with rate 1 and branches of
     * length 1, the root state r has the posterior ∝ P_{rA}(1) P_{rC}(1). So r is A or C
     * with probability p= / (2 (p= + p≠)) each, where p= and p≠ are the probabilities of
     * keeping and of changing to a given other state.
     */
    @Test
    public void testRootStateMatchesPosterior() {
        Randomizer.setSeed(1);

        Alignment alignment = new Alignment();
        alignment.initByName("sequence", List.of(
                new Sequence("A", "A"), new Sequence("B", "C")
        ), "dataType", "nucleotide");

        TreeParser tree = new TreeParser();
        tree.initByName("newick", "(A:1.0,B:1.0);", "IsLabelledNewick", true, "adjustTipHeights", false);

        Mutations mutations = new Mutations();
        mutations.initByName("tree", tree, "alignment", alignment);

        // the root sequence is the reference A, and B gains C at height 0.5

        Node b = this.getNodeByTaxon(tree, "B");
        List<List<Mutation>> history = this.createEmptyHistory(tree);
        history.get(b.getNr()).add(new Mutation(b.getNr(), 0.5, 1.0, 0, A, C));
        mutations.initialiseMutations(history);

        GeneticPrior geneticPrior = this.createGeneticPrior(alignment, tree, mutations);
        BranchReformOperator operator = new BranchReformOperator();
        operator.initByName("weight", 1.0, "mutations", mutations, "geneticPrior", geneticPrior);

        // run the chain and count the root states

        Node root = tree.getRoot();
        int[] rootStateCounts = new int[4];

        this.runChain(tree, mutations, geneticPrior, operator,
                () -> rootStateCounts[MutationPaths.getState(mutations, root, 0)]++);

        double stayProbability = 0.25 * (1.0 + 3.0 * Math.exp(-4.0 / 3.0));
        double changeProbability = 0.25 * (1.0 - Math.exp(-4.0 / 3.0));
        double expectedFrequency = stayProbability / (2.0 * (stayProbability + changeProbability));

        assertEquals(expectedFrequency, (double) rootStateCounts[A] / NUM_SAMPLES, 0.01);
        assertEquals(expectedFrequency, (double) rootStateCounts[C] / NUM_SAMPLES, 0.01);
    }

    /**
     * On the tree ((A,B)X,C)R, the branch above A carries A -> G -> T at site 0 and C -> A at
     * site 1. Under Jukes-Cantor every state leaves at the same rate, so the genetic prior
     * does not depend on the times on that branch and they are uniform under the ordering
     * constraints: the two mutations at site 0 follow the order statistics of two uniform
     * draws on the branch, and the mutation at site 1 is uniform on the branch. The other
     * branches are resampled as well, but this does not affect the branch above A.
     */
    @Test
    public void testInternalBranchTimesAreUniform() {
        Randomizer.setSeed(1);

        // B is the first taxon, so its sequence is the reference and the root sequence

        Alignment alignment = new Alignment();
        alignment.initByName("sequence", List.of(
                new Sequence("B", "AC"), new Sequence("A", "TA"), new Sequence("C", "AC")
        ), "dataType", "nucleotide");

        TreeParser tree = new TreeParser();
        tree.initByName("newick", "((A:1.0,B:1.0):1.0,C:2.0);", "IsLabelledNewick", true, "adjustTipHeights", false);

        Mutations mutations = new Mutations();
        mutations.initByName("tree", tree, "alignment", alignment);

        Node a = this.getNodeByTaxon(tree, "A");
        int nodeNr = a.getNr();

        List<List<Mutation>> history = this.createEmptyHistory(tree);
        history.get(nodeNr).addAll(List.of(
                new Mutation(nodeNr, 0.7, 1.0, 0, A, G),
                new Mutation(nodeNr, 0.5, 1.0, 1, C, A),
                new Mutation(nodeNr, 0.3, 0.7, 0, G, T)
        ));
        mutations.initialiseMutations(history);

        GeneticPrior geneticPrior = this.createGeneticPrior(alignment, tree, mutations);
        BranchReformOperator operator = new BranchReformOperator();
        operator.initByName("weight", 1.0, "mutations", mutations, "geneticPrior", geneticPrior);

        // run the chain and average the times on the branch above A

        double[] sumTimes = new double[3];

        this.runChain(tree, mutations, geneticPrior, operator, () -> {
            List<Mutation> branchMutations = mutations.getMutations(a);
            int siteZeroCount = 0;
            for (Mutation mutation : branchMutations) {
                if (mutation.site() == 0) {
                    sumTimes[siteZeroCount++] += mutation.time();
                } else {
                    sumTimes[2] += mutation.time();
                }
            }
        });

        assertEquals(2.0 / 3.0, sumTimes[0] / NUM_SAMPLES, 0.01);
        assertEquals(1.0 / 3.0, sumTimes[1] / NUM_SAMPLES, 0.01);
        assertEquals(0.5, sumTimes[2] / NUM_SAMPLES, 0.01);
    }

    /**
     * Runs a Metropolis-Hastings chain with the operator against the genetic prior on a
     * fixed tree and calls the given recorder once per sample after the burn-in.
     */
    private void runChain(TreeParser tree, Mutations mutations, GeneticPrior geneticPrior,
                          BranchReformOperator operator, Runnable recorder) {
        double logTarget = geneticPrior.calculateLogP();

        for (int i = 0; i < NUM_BURNIN + NUM_SAMPLES; i++) {
            List<List<Mutation>> oldMutations = new ArrayList<>();
            for (Node node : tree.getNodesAsArray()) {
                oldMutations.add(mutations.getMutations(node));
            }

            double logHastingsRatio = operator.proposal();

            if (logHastingsRatio != Double.NEGATIVE_INFINITY) {
                double newLogTarget = geneticPrior.calculateLogP();

                if (Math.log(Randomizer.nextDouble()) < newLogTarget - logTarget + logHastingsRatio) {
                    logTarget = newLogTarget;
                } else {
                    for (Node node : tree.getNodesAsArray()) {
                        mutations.applyMutations(node, oldMutations.get(node.getNr()), operator);
                    }
                    geneticPrior.calculateLogP();
                }
            }

            if (i >= NUM_BURNIN) {
                recorder.run();
            }
        }
    }

    private GeneticPrior createGeneticPrior(Alignment alignment, TreeParser tree, Mutations mutations) {
        SiteModel siteModel = new SiteModel();
        siteModel.initByName("substModel", new JukesCantor());

        StrictClockModel clockModel = new StrictClockModel();
        clockModel.initByName("clock.rate", new RealScalarParam<>(1.0, PositiveReal.INSTANCE));

        GeneticPrior geneticPrior = new GeneticPrior();
        geneticPrior.initByName("data", alignment, "tree", tree, "siteModel", siteModel, "branchRateModel", clockModel, "mutations", mutations);
        return geneticPrior;
    }

    private List<List<Mutation>> createEmptyHistory(TreeParser tree) {
        List<List<Mutation>> history = new ArrayList<>();
        for (int i = 0; i < tree.getNodeCount(); i++) {
            history.add(new ArrayList<>());
        }
        return history;
    }

    private Node getNodeByTaxon(TreeParser tree, String taxon) {
        for (Node node : tree.getExternalNodes()) {
            if (node.getID().equals(taxon)) {
                return node;
            }
        }
        throw new IllegalArgumentException("Unknown taxon " + taxon + ".");
    }

}
