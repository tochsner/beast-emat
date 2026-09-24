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
import emat.initalisation.ParsimonyMutationsInitialiser;
import emat.operators.IntervalScaleOperator;
import emat.operators.SiteHistoryGibbsOperator;
import emat.prior.GeneticPrior;
import emat.state.Mutation;
import emat.state.Mutations;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class IntervalScaleOperatorTest {

    private static final String SEQUENCE_A = "ACG";
    private static final String SEQUENCE_B = "CCT";

    private static final int NUM_BURNIN = 1000;
    private static final int NUM_SAMPLES = 100000;

    /** The tree, its mutations, the genetic prior and the operators of a two-tip test chain. */
    private record TwoTipChain(TreeParser tree, Mutations mutations, GeneticPrior geneticPrior,
                               IntervalScaleOperator intervalScaleOperator, SiteHistoryGibbsOperator gibbsOperator) {
    }

    /**
     * On a two-tip tree with an exponential prior on the root height h, runs a Metropolis-
     * Hastings chain that alternates the interval scale operator with the site history Gibbs
     * operator. The interval scale operator keeps the history and only maps the mutations
     * along with the branches, so the Gibbs operator is needed to resample the histories.
     * Together, they target p(h) ∝ exp(-h) L(h) with the Jukes-Cantor likelihood
     * L(h) = Π_sites P_{AB}(2h), whose mean is compared with the sampled mean.
     */
    @Test
    public void testTwoTipsMatchExactPosterior() {
        Randomizer.setSeed(1);
        TwoTipChain chain = this.createTwoTipChain();

        double meanRootHeight = this.runChain(chain, true);

        // compare with the exact posterior mean, integrated with Simpson's rule

        int numIntervals = 20000;
        double maxHeight = 40.0;
        double stepSize = maxHeight / numIntervals;

        double normalisation = 0.0;
        double expectedRootHeight = 0.0;

        for (int i = 0; i <= numIntervals; i++) {
            double height = i * stepSize;

            double density = Math.exp(-height);
            for (int site = 0; site < SEQUENCE_A.length(); site++) {
                boolean isSame = SEQUENCE_A.charAt(site) == SEQUENCE_B.charAt(site);
                density *= this.getJukesCantorProbability(isSame, 2.0 * height);
            }

            double simpsonWeight = (i == 0 || i == numIntervals) ? 1.0 : (i % 2 == 1 ? 4.0 : 2.0);
            normalisation += simpsonWeight * density;
            expectedRootHeight += simpsonWeight * height * density;
        }
        expectedRootHeight /= normalisation;

        assertEquals(expectedRootHeight, meanRootHeight, 0.03);
    }

    /**
     * Runs the same chain with only the interval scale operator, which keeps the parsimony
     * history: one mutation at each of the two differing sites. Under Jukes-Cantor with
     * rate 1, every site leaves its state at rate 1, so the genetic prior is
     * exp(-L 2h) (1/3)^M up to the mutation times. The operator scales the M mutation times
     * with the branches, whose Jacobian contributes h^M, so the root height follows
     * p(h) ∝ exp(-h) exp(-2Lh) h^M, a Gamma(M + 1, 1 + 2L) distribution with mean
     * (M + 1) / (1 + 2L) = 3/7.
     */
    @Test
    public void testFixedHistoryMatchesGammaPosterior() {
        Randomizer.setSeed(1);
        TwoTipChain chain = this.createTwoTipChain();

        double meanRootHeight = this.runChain(chain, false);

        int numSites = SEQUENCE_A.length();
        int numMutations = 2;
        double expectedRootHeight = (numMutations + 1.0) / (1.0 + 2.0 * numSites);

        assertEquals(expectedRootHeight, meanRootHeight, 0.01);
    }

    /** Sets up the two-tip tree with its parsimony history under Jukes-Cantor with rate 1. */
    private TwoTipChain createTwoTipChain() {
        Alignment alignment = new Alignment();
        alignment.initByName("sequence", List.of(new Sequence("A", SEQUENCE_A), new Sequence("B", SEQUENCE_B)), "dataType", "nucleotide");

        TreeParser tree = new TreeParser();
        tree.initByName("newick", "(A:1.0,B:1.0);", "IsLabelledNewick", true, "adjustTipHeights", false);

        Mutations mutations = new Mutations();
        mutations.initByName("tree", tree, "alignment", alignment);

        ParsimonyMutationsInitialiser initialiser = new ParsimonyMutationsInitialiser();
        initialiser.initByName("mutations", mutations);
        initialiser.initStateNodes();

        SiteModel siteModel = new SiteModel();
        siteModel.initByName("substModel", new JukesCantor());

        StrictClockModel clockModel = new StrictClockModel();
        clockModel.initByName("clock.rate", new RealScalarParam<>(1.0, PositiveReal.INSTANCE));

        GeneticPrior geneticPrior = new GeneticPrior();
        geneticPrior.initByName("data", alignment, "tree", tree, "siteModel", siteModel, "branchRateModel", clockModel, "mutations", mutations);

        IntervalScaleOperator intervalScaleOperator = new IntervalScaleOperator();
        intervalScaleOperator.initByName("weight", 1.0, "tree", tree, "mutations", mutations, "scaleFactor", 0.5, "optimise", false);

        SiteHistoryGibbsOperator gibbsOperator = new SiteHistoryGibbsOperator();
        gibbsOperator.initByName("weight", 1.0, "mutations", mutations, "geneticPrior", geneticPrior);

        return new TwoTipChain(tree, mutations, geneticPrior, intervalScaleOperator, gibbsOperator);
    }

    /**
     * Runs a Metropolis-Hastings chain on the root height with an exponential prior and the
     * genetic prior as the target, optionally resampling a site history before every
     * interval scale proposal. Returns the mean root height after the burn-in.
     */
    private double runChain(TwoTipChain chain, boolean isResamplingHistories) {
        TreeParser tree = chain.tree();
        Mutations mutations = chain.mutations();
        GeneticPrior geneticPrior = chain.geneticPrior();
        IntervalScaleOperator operator = chain.intervalScaleOperator();

        Node root = tree.getRoot();
        double logTarget = -root.getHeight() + geneticPrior.calculateLogP();
        double sumRootHeights = 0.0;

        for (int i = 0; i < NUM_BURNIN + NUM_SAMPLES; i++) {
            if (isResamplingHistories) {
                // the Gibbs operator draws from the exact conditional, so it is always accepted

                chain.gibbsOperator().proposal();
                logTarget = -root.getHeight() + geneticPrior.calculateLogP();
            }

            // scale the tree and its mutations

            double oldRootHeight = root.getHeight();
            List<List<Mutation>> oldMutations = new ArrayList<>();
            for (Node node : tree.getNodesAsArray()) {
                oldMutations.add(mutations.getMutations(node));
            }

            double logHastingsRatio = operator.proposal();
            double newLogTarget = -root.getHeight() + geneticPrior.calculateLogP();

            if (Math.log(Randomizer.nextDouble()) < newLogTarget - logTarget + logHastingsRatio) {
                logTarget = newLogTarget;
            } else {
                root.setHeight(oldRootHeight);
                for (Node node : tree.getNodesAsArray()) {
                    mutations.applyMutations(node, oldMutations.get(node.getNr()), operator);
                }
            }

            if (i >= NUM_BURNIN) {
                sumRootHeights += root.getHeight();
            }
        }

        return sumRootHeights / NUM_SAMPLES;
    }

    /** Returns the Jukes-Cantor probability of ending in the same or a given other state over the given time. */
    private double getJukesCantorProbability(boolean isSame, double time) {
        double decay = Math.exp(-4.0 * time / 3.0);
        return isSame ? 0.25 + 0.75 * decay : 0.25 - 0.25 * decay;
    }

}
