package emat.operators.nodetimes;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeParser;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.evolution.branchratemodel.StrictClockModel;
import beast.base.evolution.substitutionmodel.SubstitutionModel;
import beast.base.spec.evolution.sitemodel.SiteModel;
import beast.base.spec.evolution.substitutionmodel.JukesCantor;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.util.Randomizer;
import emat.initalisation.ParsimonyMutationsInitialiser;
import emat.prior.GeneticPrior;
import emat.state.Mutation;
import emat.state.Mutations;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class RootScaleOperatorTest {

    private static final String SEQUENCE_A = "ACG";
    private static final String SEQUENCE_B = "CCT";

    private static final int NUM_BURNIN = 1000;
    private static final int NUM_SAMPLES = 100000;

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
        operator.initByName("weight", 1.0, "tree", setup.tree(), "mutations", setup.mutations(), "scaleFactor", 0.5, "optimise", false);

        int numSites = SEQUENCE_A.length();
        int numMutations = 2;
        double expectedRootHeight = (numMutations + 1.0) / (1.0 + 2.0 * numSites);

        assertEquals(expectedRootHeight, this.runChain(setup, operator, NUM_SAMPLES), 0.01);
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
