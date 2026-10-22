package emat.operators.mutations;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.substitutionmodel.SubstitutionModel;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeParser;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.evolution.branchratemodel.StrictClockModel;
import beast.base.spec.evolution.sitemodel.SiteModel;
import beast.base.spec.evolution.substitutionmodel.Frequencies;
import beast.base.spec.evolution.substitutionmodel.HKY;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.spec.inference.parameter.SimplexParam;
import beast.base.util.Randomizer;
import emat.helper.MutationPaths;
import emat.initalisation.ParsimonyMutationsInitialiser;
import emat.prior.GeneticPrior;
import emat.state.Mutation;
import emat.state.Mutations;
import emat.stochasticmapping.StochasticMapping;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class MissingCladeGibbsOperatorTest {

    // on ((A,B)X,C)R, site 0 is missing at A and B, so the clade X is missing; site 1 is missing at A only
    private static final List<Sequence> SEQUENCES = List.of(new Sequence("A", "NN"), new Sequence("B", "NC"), new Sequence("C", "GC"));
    private static final String NEWICK = "((A:0.3,B:0.5)X:0.4,C:0.9)R;";

    /**
     * As only this operator runs, the root state r at site 0 stays fixed, while the whole
     * clade X is resampled: the states of X, A and B must follow P_r·(0.4), P_r·(0.7) and
     * P_r·(0.9). At site 1, the clade stops at A, whose sibling B is observed, so the state
     * x of X stays fixed and A must follow P_x·(0.3).
     */
    @Test
    public void testMissingCladeFollowsForwardProcess() {
        Randomizer.setSeed(1);
        TestSetup setup = new TestSetup(Integer.MAX_VALUE);

        Node tipA = setup.getNode("A");
        Node tipB = setup.getNode("B");
        Node inner = tipA.getParent();
        int rootState = MutationPaths.getState(setup.mutations, setup.tree.getRoot(), 0);
        int innerStateAtSite1 = MutationPaths.getState(setup.mutations, inner, 1);
        List<Mutation> observedMutations = SiteMutations.collect(setup.mutations, tipB, 1);

        // run the operator and collect the states at both sites

        int numSamples = 400000;
        double[][] stateCounts = new double[4][4];
        for (int i = 0; i < numSamples; i++) {
            assertEquals(Double.POSITIVE_INFINITY, setup.operator.proposal());
            setup.mutations.setEverythingDirty(false);

            stateCounts[0][MutationPaths.getState(setup.mutations, inner, 0)]++;
            stateCounts[1][MutationPaths.getState(setup.mutations, tipA, 0)]++;
            stateCounts[2][MutationPaths.getState(setup.mutations, tipB, 0)]++;
            stateCounts[3][MutationPaths.getState(setup.mutations, tipA, 1)]++;
        }

        assertEquals(innerStateAtSite1, MutationPaths.getState(setup.mutations, inner, 1), "The state of X at site 1 changed.");
        assertEquals(observedMutations, SiteMutations.collect(setup.mutations, tipB, 1), "The observed tip B changed at site 1.");

        // compare with the forward process

        double[][] expected = {
                setup.getTransitionProbabilities(0.4), setup.getTransitionProbabilities(0.7),
                setup.getTransitionProbabilities(0.9), setup.getTransitionProbabilities(0.3),
        };
        int[] startStates = {rootState, rootState, rootState, innerStateAtSite1};
        String[] names = {"X at site 0", "A at site 0", "B at site 0", "A at site 1"};

        for (int k = 0; k < 4; k++) {
            for (int state = 0; state < 4; state++) {
                assertEquals(expected[k][startStates[k] * 4 + state], stateCounts[k][state] / numSamples, 0.005,
                        names[k] + ", state " + state + ".");
            }
        }
    }

    /** Without any levels, only the branches above the picked tips are resampled, so the state of X never changes. */
    @Test
    public void testWithoutLevelsKeepsInnerStates() {
        Randomizer.setSeed(2);
        TestSetup setup = new TestSetup(0);

        Node inner = setup.getNode("A").getParent();
        int innerState = MutationPaths.getState(setup.mutations, inner, 0);
        int numChangedTipStates = 0;

        for (int i = 0; i < 20000; i++) {
            int oldTipState = MutationPaths.getState(setup.mutations, setup.getNode("B"), 0);
            setup.operator.proposal();
            setup.mutations.setEverythingDirty(false);

            assertEquals(innerState, MutationPaths.getState(setup.mutations, inner, 0));
            if (MutationPaths.getState(setup.mutations, setup.getNode("B"), 0) != oldTipState) {
                numChangedTipStates++;
            }
        }

        assertEquals(true, numChangedTipStates > 0, "The operator never changed the state of a missing tip.");
    }

    /* Test Setup */

    /**
     * Builds the tree, alignment, mutations (initialised by parsimony), a genetic prior under
     * HKY with clock rate 1, and the operator with the given maximum number of levels.
     */
    private static class TestSetup {

        final TreeParser tree;
        final Mutations mutations;
        final SubstitutionModel substitutionModel;
        final MissingCladeGibbsOperator operator;

        TestSetup(int maxLevels) {
            Alignment alignment = new Alignment();
            alignment.initByName("sequence", SEQUENCES, "dataType", "nucleotide");

            this.tree = new TreeParser();
            this.tree.initByName("newick", NEWICK, "IsLabelledNewick", true, "adjustTipHeights", false);

            this.mutations = new Mutations();
            this.mutations.initByName("tree", this.tree, "alignment", alignment);

            ParsimonyMutationsInitialiser initialiser = new ParsimonyMutationsInitialiser();
            initialiser.initByName("mutations", this.mutations);
            initialiser.initStateNodes();

            Frequencies frequencies = new Frequencies();
            frequencies.initByName("frequencies", new SimplexParam(new double[]{0.1, 0.2, 0.3, 0.4}));
            HKY hky = new HKY();
            hky.initByName("kappa", new RealScalarParam<>(4.0, PositiveReal.INSTANCE), "frequencies", frequencies);
            this.substitutionModel = hky;

            SiteModel siteModel = new SiteModel();
            siteModel.initByName("substModel", hky);

            StrictClockModel clockModel = new StrictClockModel();
            clockModel.initByName("clock.rate", new RealScalarParam<>(1.0, PositiveReal.INSTANCE));

            GeneticPrior geneticPrior = new GeneticPrior();
            geneticPrior.initByName(
                    "data", alignment,
                    "tree", this.tree,
                    "siteModel", siteModel,
                    "branchRateModel", clockModel,
                    "mutations", this.mutations
            );

            this.operator = new MissingCladeGibbsOperator();
            this.operator.initByName("weight", 1.0, "mutations", this.mutations, "geneticPrior", geneticPrior, "maxLevels", maxLevels);
        }

        Node getNode(String taxon) {
            for (Node node : this.tree.getExternalNodes()) {
                if (node.getID().equals(taxon)) {
                    return node;
                }
            }
            throw new IllegalArgumentException("Unknown taxon " + taxon);
        }

        double[] getTransitionProbabilities(double time) {
            return StochasticMapping.computeTransitionProbabilities(this.substitutionModel.getEigenDecomposition(null), time);
        }

    }

}
