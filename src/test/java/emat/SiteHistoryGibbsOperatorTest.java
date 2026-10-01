package emat;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.substitutionmodel.SubstitutionModel;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeParser;
import beast.base.inference.State;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.evolution.branchratemodel.StrictClockModel;
import beast.base.spec.evolution.sitemodel.SiteModel;
import beast.base.spec.evolution.substitutionmodel.Frequencies;
import beast.base.spec.evolution.substitutionmodel.HKY;
import beast.base.spec.evolution.substitutionmodel.JukesCantor;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.spec.inference.parameter.SimplexParam;
import beast.base.util.Randomizer;
import emat.stochasticmapping.StochasticMapping;
import emat.initalisation.ParsimonyMutationsInitialiser;
import emat.operators.SiteHistoryGibbsOperator;
import emat.prior.GeneticPrior;
import emat.state.Mutation;
import emat.state.Mutations;
import org.junit.jupiter.api.Test;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class SiteHistoryGibbsOperatorTest {

    // the length of both branches of the two-tip tree
    private static final double BRANCH_LENGTH = 0.8;

    /**
     * On a two-tip tree with tips A and C under Jukes-Cantor, checks the sampled root state
     * against its exact posterior P(r) ∝ π_r P_rA(t) P_rC(t), and the sampled number of
     * mutations against its exact expectation.
     */
    @Test
    public void testTwoTipsMatchExactPosterior() {
        Randomizer.setSeed(1);

        TestSetup setup = new TestSetup(
                List.of(new Sequence("A", "A"), new Sequence("B", "C")),
                "(A:" + BRANCH_LENGTH + ",B:" + BRANCH_LENGTH + ");"
        );

        // run the operator and collect the root states and the numbers of mutations

        int numBurnin = 1000;
        int numSamples = 200000;

        int[] rootStateCounts = new int[4];
        double sumNumMutations = 0.0;

        for (int i = 0; i < numBurnin + numSamples; i++) {
            setup.operator.proposal();

            if (i < numBurnin) {
                continue;
            }

            rootStateCounts[setup.getRootState(0)]++;
            for (Node node : setup.tree.getNodesAsArray()) {
                if (!node.isRoot()) {
                    sumNumMutations += setup.mutations.getMutations(node).size();
                }
            }
        }

        // compare with the exact posterior

        double[] rootProbabilities = new double[4];
        double expectedNumMutations = 0.0;
        double normalisation = 0.0;

        for (int rootState = 0; rootState < 4; rootState++) {
            double weight = 0.25
                    * this.getJukesCantorProbability(rootState, 0, BRANCH_LENGTH)
                    * this.getJukesCantorProbability(rootState, 1, BRANCH_LENGTH);

            rootProbabilities[rootState] = weight;
            normalisation += weight;
            expectedNumMutations += weight * (
                    this.getExpectedNumMutations(rootState, 0, BRANCH_LENGTH)
                            + this.getExpectedNumMutations(rootState, 1, BRANCH_LENGTH)
            );
        }
        expectedNumMutations /= normalisation;

        for (int rootState = 0; rootState < 4; rootState++) {
            assertEquals(rootProbabilities[rootState] / normalisation, (double) rootStateCounts[rootState] / numSamples, 0.005);
        }
        assertEquals(expectedNumMutations, sumNumMutations / numSamples, 0.01);
    }

    /**
     * On a larger tree with ambiguous and missing characters, checks after every move that
     * replaying the mutations from the reference reproduces a state compatible with every
     * tip. Invalid mutation lists are also caught by the sanity checks in Mutations.
     */
    @Test
    public void testHistoriesStayConsistentWithTips() {
        Randomizer.setSeed(2);

        TestSetup setup = new TestSetup(
                List.of(
                        new Sequence("A", "ACGTR"),
                        new Sequence("B", "ACGAN"),
                        new Sequence("C", "TCRAA"),
                        new Sequence("D", "TCGT-"),
                        new Sequence("E", "GCGTA")
                ),
                "(((A:0.3,B:0.3):0.4,(C:0.5,D:0.5):0.2):0.3,E:1.0);"
        );

        for (int i = 0; i < 5000; i++) {
            setup.operator.proposal();

            for (Node tip : setup.tree.getExternalNodes()) {
                int taxonNr = setup.alignment.getTaxonIndex(tip.getID());
                int[] tipSequence = setup.getSequence(tip);

                for (int site = 0; site < tipSequence.length; site++) {
                    int code = setup.alignment.getPattern(taxonNr, setup.alignment.getPatternIndex(site));
                    int tipState = tipSequence[site];

                    boolean isCompatible = false;
                    for (int state : setup.alignment.getDataType().getStatesForCode(code)) {
                        isCompatible |= state == tipState;
                    }
                    assertTrue(isCompatible, "Tip " + tip.getID() + " is inconsistent at site " + site + ".");
                }
            }
        }
    }

    /**
     * With a Jukes-Cantor genetic prior, the Jukes-Cantor sampler maps under the real model,
     * so the Hastings ratio of every move must cancel the change of the genetic prior.
     */
    @Test
    public void testJukesCantorSamplerIsExactUnderJukesCantor() {
        Randomizer.setSeed(3);

        TestSetup setup = new TestSetup(
                List.of(
                        new Sequence("A", "ACGTR"),
                        new Sequence("B", "ACGAN"),
                        new Sequence("C", "TCRAA"),
                        new Sequence("D", "TCGT-"),
                        new Sequence("E", "GCGTA")
                ),
                "(((A:0.3,B:0.3):0.4,(C:0.5,D:0.5):0.2):0.3,E:1.0);",
                new JukesCantor(), SiteHistoryGibbsOperator.Sampler.JUKES_CANTOR
        );

        double oldLogP = setup.computeLogGeneticPrior();
        for (int i = 0; i < 2000; i++) {
            double logHastingsRatio = setup.operator.proposal();
            double newLogP = setup.computeLogGeneticPrior();

            assertEquals(0.0, newLogP - oldLogP + logHastingsRatio, 1e-9);
            oldLogP = newLogP;
        }
    }

    /**
     * On a two-tip tree under HKY, where the Jukes-Cantor sampler only approximates the
     * model, runs the operator with Metropolis-Hastings acceptance and checks the sampled
     * root state and number of mutations against their exact posterior values.
     */
    @Test
    public void testJukesCantorSamplerMatchesExactPosteriorUnderHky() {
        Randomizer.setSeed(4);

        Frequencies frequencies = new Frequencies();
        frequencies.initByName("frequencies", new SimplexParam(new double[]{0.1, 0.2, 0.3, 0.4}));
        HKY hky = new HKY();
        hky.initByName("kappa", new RealScalarParam<>(4.0, PositiveReal.INSTANCE), "frequencies", frequencies);

        TestSetup setup = new TestSetup(
                List.of(new Sequence("A", "A"), new Sequence("B", "C")),
                "(A:" + BRANCH_LENGTH + ",B:" + BRANCH_LENGTH + ");",
                hky, SiteHistoryGibbsOperator.Sampler.JUKES_CANTOR
        );

        State state = new State();
        state.initByName("stateNode", setup.tree, "stateNode", setup.mutations);
        state.initialise();
        state.setPosterior(setup.geneticPrior);
        state.setEverythingDirty(true);
        double oldLogP = setup.geneticPrior.calculateLogP();
        state.setEverythingDirty(false);

        // run the operator with Metropolis-Hastings acceptance and collect the root states and the numbers of mutations

        int numBurnin = 1000;
        int numSamples = 200000;

        int[] rootStateCounts = new int[4];
        double sumNumMutations = 0.0;

        for (int i = 0; i < numBurnin + numSamples; i++) {
            state.store(i);
            double logHastingsRatio = setup.operator.proposal();

            state.storeCalculationNodes();
            state.checkCalculationNodesDirtiness();
            double newLogP = setup.geneticPrior.calculateLogP();

            if (Math.log(Randomizer.nextDouble()) < newLogP - oldLogP + logHastingsRatio) {
                oldLogP = newLogP;
                state.acceptCalculationNodes();
            } else {
                state.restore();
                state.restoreCalculationNodes();
            }
            state.setEverythingDirty(false);

            if (i < numBurnin) {
                continue;
            }

            rootStateCounts[setup.getRootState(0)]++;
            for (Node node : setup.tree.getNodesAsArray()) {
                if (!node.isRoot()) {
                    sumNumMutations += setup.mutations.getMutations(node).size();
                }
            }
        }

        // compare with the exact posterior

        double[] rateMatrix = setup.geneticPrior.computeRateMatrix();
        double[] transitionProbabilities = StochasticMapping.computeTransitionProbabilities(hky.getEigenDecomposition(null), BRANCH_LENGTH);
        double[] stationaryFrequencies = hky.getFrequencies();

        double[] rootProbabilities = new double[4];
        double expectedNumMutations = 0.0;
        double normalisation = 0.0;

        for (int rootState = 0; rootState < 4; rootState++) {
            double weight = stationaryFrequencies[rootState]
                    * transitionProbabilities[rootState * 4]
                    * transitionProbabilities[rootState * 4 + 1];

            rootProbabilities[rootState] = weight;
            normalisation += weight;
            expectedNumMutations += weight * (
                    this.getExpectedNumMutations(hky, rateMatrix, rootState, 0, BRANCH_LENGTH)
                            + this.getExpectedNumMutations(hky, rateMatrix, rootState, 1, BRANCH_LENGTH)
            );
        }
        expectedNumMutations /= normalisation;

        for (int rootState = 0; rootState < 4; rootState++) {
            assertEquals(rootProbabilities[rootState] / normalisation, (double) rootStateCounts[rootState] / numSamples, 0.005);
        }
        assertEquals(expectedNumMutations, sumNumMutations / numSamples, 0.01);
    }

    /* General Reference Values */

    /**
     * Returns the expected number of mutations on a branch of the given length, conditional
     * on its end states, under the given substitution model with rate matrix Q:
     * E[N] = ∫ Σ_{c≠d} P_ac(s) Q_cd P_db(t - s) ds / P_ab(t), integrated with Simpson's rule.
     */
    private double getExpectedNumMutations(SubstitutionModel substitutionModel, double[] rateMatrix,
                                           int startState, int endState, double time) {
        int numIntervals = 2000;
        double stepSize = time / numIntervals;

        double integral = 0.0;
        for (int i = 0; i <= numIntervals; i++) {
            double s = i * stepSize;
            double[] startProbabilities = StochasticMapping.computeTransitionProbabilities(substitutionModel.getEigenDecomposition(null), s);
            double[] endProbabilities = StochasticMapping.computeTransitionProbabilities(substitutionModel.getEigenDecomposition(null), time - s);

            double integrand = 0.0;
            for (int c = 0; c < 4; c++) {
                for (int d = 0; d < 4; d++) {
                    if (c != d) {
                        integrand += startProbabilities[startState * 4 + c] * rateMatrix[c * 4 + d] * endProbabilities[d * 4 + endState];
                    }
                }
            }

            double simpsonWeight = (i == 0 || i == numIntervals) ? 1.0 : (i % 2 == 1 ? 4.0 : 2.0);
            integral += simpsonWeight * integrand;
        }
        integral *= stepSize / 3.0;

        double[] transitionProbabilities = StochasticMapping.computeTransitionProbabilities(substitutionModel.getEigenDecomposition(null), time);
        return integral / transitionProbabilities[startState * 4 + endState];
    }

    /* Jukes-Cantor Reference Values */

    /** Returns the Jukes-Cantor transition probability from one state to another over the given time. */
    private double getJukesCantorProbability(int from, int to, double time) {
        double decay = Math.exp(-4.0 * time / 3.0);
        return from == to ? 0.25 + 0.75 * decay : 0.25 - 0.25 * decay;
    }

    /**
     * Returns the expected number of mutations on a branch of the given length, conditional
     * on its end states, under Jukes-Cantor:
     * E[N] = ∫ Σ_{c≠d} P_ac(s) Q_cd P_db(t - s) ds / P_ab(t), integrated with Simpson's rule.
     */
    private double getExpectedNumMutations(int startState, int endState, double time) {
        int numIntervals = 2000;
        double stepSize = time / numIntervals;

        double integral = 0.0;
        for (int i = 0; i <= numIntervals; i++) {
            double s = i * stepSize;

            double integrand = 0.0;
            for (int c = 0; c < 4; c++) {
                for (int d = 0; d < 4; d++) {
                    if (c != d) {
                        integrand += this.getJukesCantorProbability(startState, c, s) / 3.0
                                * this.getJukesCantorProbability(d, endState, time - s);
                    }
                }
            }

            double simpsonWeight = (i == 0 || i == numIntervals) ? 1.0 : (i % 2 == 1 ? 4.0 : 2.0);
            integral += simpsonWeight * integrand;
        }
        integral *= stepSize / 3.0;

        return integral / this.getJukesCantorProbability(startState, endState, time);
    }

    /* Test Setup */

    /**
     * Builds the tree, alignment, mutations (initialised by parsimony), a genetic prior with
     * the given substitution model and clock rate 1, and the operator with the given sampler.
     */
    private static class TestSetup {

        final Alignment alignment;
        final TreeParser tree;
        final Mutations mutations;
        final GeneticPrior geneticPrior;
        final SiteHistoryGibbsOperator operator;

        TestSetup(List<Sequence> sequences, String newick) {
            this(sequences, newick, new JukesCantor(), SiteHistoryGibbsOperator.Sampler.EXACT);
        }

        TestSetup(List<Sequence> sequences, String newick, SubstitutionModel substitutionModel, SiteHistoryGibbsOperator.Sampler sampler) {
            this.alignment = new Alignment();
            this.alignment.initByName("sequence", sequences, "dataType", "nucleotide");

            this.tree = new TreeParser();
            this.tree.initByName("newick", newick, "IsLabelledNewick", true, "adjustTipHeights", false);

            this.mutations = new Mutations();
            this.mutations.initByName("tree", this.tree, "alignment", this.alignment);

            ParsimonyMutationsInitialiser initialiser = new ParsimonyMutationsInitialiser();
            initialiser.initByName("mutations", this.mutations);
            initialiser.initStateNodes();

            SiteModel siteModel = new SiteModel();
            siteModel.initByName("substModel", substitutionModel);

            StrictClockModel clockModel = new StrictClockModel();
            clockModel.initByName("clock.rate", new RealScalarParam<>(1.0, PositiveReal.INSTANCE));

            this.geneticPrior = new GeneticPrior();
            this.geneticPrior.initByName(
                    "data", this.alignment,
                    "tree", this.tree,
                    "siteModel", siteModel,
                    "branchRateModel", clockModel,
                    "mutations", this.mutations
            );

            this.operator = new SiteHistoryGibbsOperator();
            this.operator.initByName("weight", 1.0, "mutations", this.mutations, "geneticPrior", this.geneticPrior, "sampler", sampler);
        }

        /** Computes the genetic prior of the current state from scratch with a new genetic prior on the same inputs. */
        double computeLogGeneticPrior() {
            GeneticPrior freshGeneticPrior = new GeneticPrior();
            freshGeneticPrior.initByName(
                    "data", this.alignment,
                    "tree", this.tree,
                    "siteModel", this.geneticPrior.siteModelInput.get(),
                    "branchRateModel", this.geneticPrior.branchRateModelInput.get(),
                    "mutations", this.mutations
            );
            return freshGeneticPrior.calculateLogP();
        }

        /** Returns the state of the given site at the root, i.e. the reference after the root mutations. */
        int getRootState(int site) {
            int state = this.mutations.getReferenceSequence()[site];
            for (Mutation mutation : this.mutations.getMutations(this.tree.getRoot())) {
                if (mutation.site() == site) {
                    state = mutation.newState();
                }
            }
            return state;
        }

        /** Returns the sequence at the given node, obtained by replaying all mutations from the reference. */
        int[] getSequence(Node node) {
            Deque<Node> path = new ArrayDeque<>();
            for (Node current = node; current != null; current = current.getParent()) {
                path.push(current);
            }

            int[] sequence = this.mutations.getReferenceSequence().clone();
            for (Node current : path) {
                for (Mutation mutation : this.mutations.getMutations(current)) {
                    assertEquals(sequence[mutation.site()], mutation.oldState(), "Mutation does not start from the current state.");
                    sequence[mutation.site()] = mutation.newState();
                }
            }
            return sequence;
        }

    }

}
