package emat.operators.nodetimes;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeParser;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.evolution.branchratemodel.StrictClockModel;
import beast.base.evolution.substitutionmodel.SubstitutionModel;
import beast.base.spec.evolution.sitemodel.SiteModel;
import beast.base.spec.evolution.substitutionmodel.Frequencies;
import beast.base.spec.evolution.substitutionmodel.HKY;
import beast.base.spec.evolution.substitutionmodel.JukesCantor;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.spec.inference.parameter.SimplexParam;
import beast.base.util.Randomizer;
import emat.initalisation.ParsimonyMutationsInitialiser;
import emat.prior.GeneticPrior;
import emat.state.Mutation;
import emat.state.Mutations;
import emat.stochasticmapping.SubstitutionProcess;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class RootScaleOperatorTest {

    private static final String SEQUENCE_A = "ACG";
    private static final String SEQUENCE_B = "CCT";

    // the tips of the hybrid history test, which differ at some sites by transitions and at others by transversions
    private static final String HYBRID_SEQUENCE_A = "ACGTAC";
    private static final String HYBRID_SEQUENCE_B = "GCATCC";

    private static final int NUM_BURNIN = 1000;
    private static final int NUM_SAMPLES = 100000;
    private static final int NUM_HYBRID_SAMPLES = 400000;

    private static final int NUM_GRID_POINTS = 20000;
    private static final double MAX_GRID_HEIGHT = 20.0;

    /**
     * On a two-tip tree with an exponential prior on the root height h, runs a Metropolis-
     * Hastings chain with only the root scale operator, which keeps the parsimony history:
     * one mutation at each of the two differing sites. Under Jukes-Cantor with rate 1, the
     * genetic prior is exp(-L 2h) (1/3)^M up to the mutation times, and scaling the M
     * mutation times with the branches contributes h^M, so the root height follows a
     * Gamma(M + 1, 1 + 2L) distribution with mean (M + 1) / (1 + 2L) = 3/7.
     */
    @Test
    public void testFixedHistoryMatchesGammaPosterior() {
        Randomizer.setSeed(1);
        Setup setup = this.createSetup(new JukesCantor(), 1.0, SEQUENCE_A, SEQUENCE_B);

        RootScaleOperator operator = new RootScaleOperator();
        operator.initByName("weight", 1.0, "tree", setup.tree(), "mutations", setup.mutations(), "historySampler", RootScaleOperator.PathHistory.LINEAR,
                "scaleFactor", 0.5, "optimise", false);

        int numSites = SEQUENCE_A.length();
        int numMutations = 2;
        double expectedRootHeight = (numMutations + 1.0) / (1.0 + 2.0 * numSites);

        assertEquals(expectedRootHeight, this.runChain(setup, operator, NUM_SAMPLES), 0.01);
    }

    /**
     * On a two-tip tree with an exponential prior on the root height h, runs a Metropolis-
     * Hastings chain with only the root scale operator and the hybrid history sampler, which
     * resamples the history of the path between the tips and the root sequence. The chain
     * then follows the posterior with the histories integrated out: under a time-reversible
     * model with frequencies π, the root and both branches contribute π_a P_ab(2 r h) at a
     * site with tip states a and b, for clock rate r. The posterior mean of h is computed on
     * a grid. HKY with unequal frequencies makes the Jukes-Cantor choice of the sites differ
     * from the model.
     */
    @Test
    public void testHybridHistoryMatchesIntegratedPosterior() {
        Randomizer.setSeed(3);

        Frequencies frequencies = new Frequencies();
        frequencies.initByName("frequencies", new SimplexParam(new double[]{0.1, 0.2, 0.3, 0.4}));
        HKY hky = new HKY();
        hky.initByName("kappa", new RealScalarParam<>(4.0, PositiveReal.INSTANCE), "frequencies", frequencies);
        Setup setup = this.createSetup(hky, 0.7, HYBRID_SEQUENCE_A, HYBRID_SEQUENCE_B);

        RootScaleOperator operator = new RootScaleOperator();
        operator.initByName("weight", 1.0, "tree", setup.tree(), "mutations", setup.mutations(), "geneticPrior", setup.geneticPrior(),
                "historySampler", RootScaleOperator.PathHistory.HYBRID, "scaleFactor", 0.5, "optimise", false);

        double expectedRootHeight = this.computeIntegratedPosteriorMean(setup, hky, HYBRID_SEQUENCE_A, HYBRID_SEQUENCE_B);
        assertEquals(expectedRootHeight, this.runChain(setup, operator, NUM_HYBRID_SAMPLES), 0.01);
    }

    /**
     * Computes the posterior mean of the root height h under the prior exp(-h) and the
     * likelihood of the two tips with the histories integrated out, on a grid.
     */
    private double computeIntegratedPosteriorMean(Setup setup, SubstitutionModel substitutionModel, String sequenceA, String sequenceB) {
        int numStates = 4;
        int[] statesA = setup.alignment().getDataType().stringToEncoding(sequenceA).stream().mapToInt(Integer::intValue).toArray();
        int[] statesB = setup.alignment().getDataType().stringToEncoding(sequenceB).stream().mapToInt(Integer::intValue).toArray();

        double sumDensities = 0.0;
        double sumWeightedHeights = 0.0;
        for (int i = 1; i <= NUM_GRID_POINTS; i++) {
            double height = MAX_GRID_HEIGHT * i / NUM_GRID_POINTS;
            double[] probabilities = SubstitutionProcess.computeTransitionProbabilities(
                    substitutionModel.getEigenDecomposition(null), 2.0 * setup.clockRate() * height
            );

            double logDensity = -height;
            for (int site = 0; site < statesA.length; site++) {
                logDensity += Math.log(probabilities[statesA[site] * numStates + statesB[site]]);
            }

            double density = Math.exp(logDensity);
            sumDensities += density;
            sumWeightedHeights += density * height;
        }

        return sumWeightedHeights / sumDensities;
    }

    /**
     * Runs a Metropolis-Hastings chain with the given operator against the genetic prior and
     * the prior exp(-h) on the root height h, and returns the mean root height after the
     * burn-in.
     */
    private double runChain(Setup setup, RootScaleOperator operator, int numSamples) {
        TreeParser tree = setup.tree();
        Mutations mutations = setup.mutations();
        GeneticPrior geneticPrior = setup.geneticPrior();

        Node root = tree.getRoot();
        double logTarget = -root.getHeight() + geneticPrior.calculateLogP();
        double sumRootHeights = 0.0;

        for (int i = 0; i < NUM_BURNIN + numSamples; i++) {
            double oldRootHeight = root.getHeight();
            List<List<Mutation>> oldMutations = new ArrayList<>();
            for (Node node : tree.getNodesAsArray()) {
                oldMutations.add(mutations.getMutations(node));
            }

            double logHastingsRatio = operator.proposal();

            if (logHastingsRatio != Double.NEGATIVE_INFINITY) {
                double newLogTarget = -root.getHeight() + geneticPrior.calculateLogP();

                if (Math.log(Randomizer.nextDouble()) < newLogTarget - logTarget + logHastingsRatio) {
                    logTarget = newLogTarget;
                } else {
                    root.setHeight(oldRootHeight);
                    for (Node node : tree.getNodesAsArray()) {
                        mutations.applyMutations(node, oldMutations.get(node.getNr()), operator);
                    }
                }
            }

            if (i >= NUM_BURNIN) {
                sumRootHeights += root.getHeight();
            }
        }

        return sumRootHeights / numSamples;
    }

    private record Setup(Alignment alignment, TreeParser tree, Mutations mutations, GeneticPrior geneticPrior, double clockRate) {
    }

    /** Creates the two-tip tree with parsimonious mutations and the genetic prior under the given model and clock rate. */
    private Setup createSetup(SubstitutionModel substitutionModel, double clockRate, String sequenceA, String sequenceB) {
        Alignment alignment = new Alignment();
        alignment.initByName("sequence", List.of(new Sequence("A", sequenceA), new Sequence("B", sequenceB)), "dataType", "nucleotide");

        TreeParser tree = new TreeParser();
        tree.initByName("newick", "(A:1.0,B:1.0);", "IsLabelledNewick", true, "adjustTipHeights", false);

        Mutations mutations = new Mutations();
        mutations.initByName("tree", tree, "alignment", alignment);

        ParsimonyMutationsInitialiser initialiser = new ParsimonyMutationsInitialiser();
        initialiser.initByName("mutations", mutations);
        initialiser.initStateNodes();

        SiteModel siteModel = new SiteModel();
        siteModel.initByName("substModel", substitutionModel);

        StrictClockModel clockModel = new StrictClockModel();
        clockModel.initByName("clock.rate", new RealScalarParam<>(clockRate, PositiveReal.INSTANCE));

        GeneticPrior geneticPrior = new GeneticPrior();
        geneticPrior.initByName("data", alignment, "tree", tree, "siteModel", siteModel, "branchRateModel", clockModel, "mutations", mutations);

        return new Setup(alignment, tree, mutations, geneticPrior, clockRate);
    }

}
