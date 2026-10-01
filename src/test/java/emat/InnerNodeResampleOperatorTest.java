package emat;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeParser;
import beast.base.inference.State;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.evolution.branchratemodel.StrictClockModel;
import beast.base.spec.evolution.sitemodel.SiteModel;
import beast.base.spec.evolution.substitutionmodel.JukesCantor;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.util.Randomizer;
import emat.helper.MutationPaths;
import emat.initalisation.ParsimonyMutationsInitialiser;
import emat.operators.InnerNodeResampleOperator;
import emat.prior.GeneticPrior;
import emat.state.Mutations;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class InnerNodeResampleOperatorTest {

    private static final int NUM_BURNIN = 20000;
    private static final int NUM_SAMPLES = 1000000;
    private static final double TOLERANCE = 0.005;
    private static final int NUM_GRID_POINTS = 400;

    private static final double ROOT_HEIGHT = 1.0;
    private static final String[] TAXA = {"A", "B", "C", "D"};
    private static final String[] SEQUENCES = {"ACGTC", "ACGAA", "TCGAC", "TGCAT"};

    /**
     * Runs the operator against the genetic prior under Jukes-Cantor on the caterpillar
     * (((A,B)X,C)Y,D)R with all tips at height 0. The operator keeps the topology, the root
     * height and the root sequence, so the heights u of X and v of Y follow the Felsenstein
     * likelihood given the root sequence on 0 < u < v < 1. The history on the branch above D
     * is never resampled, but it does not depend on u and v given the root sequence.
     */
    @Test
    public void testHeightsMatchPosterior() {
        Randomizer.setSeed(5);

        Alignment alignment = new Alignment();
        List<Sequence> sequences = new ArrayList<>();
        for (int i = 0; i < TAXA.length; i++) {
            sequences.add(new Sequence(TAXA[i], SEQUENCES[i]));
        }
        alignment.initByName("sequence", sequences, "dataType", "nucleotide");

        TreeParser tree = new TreeParser();
        tree.initByName("newick", "(((A:0.4,B:0.4):0.3,C:0.7):0.3,D:1.0);", "IsLabelledNewick", true, "adjustTipHeights", false);

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

        InnerNodeResampleOperator operator = new InnerNodeResampleOperator();
        operator.initByName("weight", 1.0, "tree", tree, "mutations", mutations, "geneticPrior", geneticPrior);

        int[] rootSequence = new int[SEQUENCES[0].length()];
        for (int site = 0; site < rootSequence.length; site++) {
            rootSequence[site] = MutationPaths.getState(mutations, tree.getRoot(), site);
        }

        // run the chain through the same store, accept and restore cycle as MCMC

        State state = new State();
        state.initByName("stateNode", tree, "stateNode", mutations);
        state.initialise();
        state.setPosterior(geneticPrior);
        state.setEverythingDirty(true);
        state.checkCalculationNodesDirtiness();

        double oldLogP = geneticPrior.calculateLogP();
        state.setEverythingDirty(false);

        double sumXHeights = 0.0;
        double sumYHeights = 0.0;

        for (long i = 0; i < NUM_BURNIN + NUM_SAMPLES; i++) {
            state.store(i);

            double logHastingsRatio = operator.proposal();
            if (logHastingsRatio == Double.NEGATIVE_INFINITY) {
                state.restore();
            } else {
                state.storeCalculationNodes();
                state.checkCalculationNodesDirtiness();

                double newLogP = geneticPrior.calculateLogP();
                if (Math.log(Randomizer.nextDouble()) < newLogP - oldLogP + logHastingsRatio) {
                    oldLogP = newLogP;
                    state.acceptCalculationNodes();
                } else {
                    state.restore();
                    state.restoreCalculationNodes();
                }
            }
            state.setEverythingDirty(false);

            if (i >= NUM_BURNIN) {
                // restoring the tree replaces its nodes, so they are looked up after every step
                Node y = this.getInternalChild(tree.getRoot());
                Node x = this.getInternalChild(y);
                sumXHeights += x.getHeight();
                sumYHeights += y.getHeight();
            }
        }

        double[] expectedHeights = this.computeExpectedHeights(alignment, rootSequence);
        assertEquals(expectedHeights[0], sumXHeights / NUM_SAMPLES, TOLERANCE);
        assertEquals(expectedHeights[1], sumYHeights / NUM_SAMPLES, TOLERANCE);
    }

    /* Exact Posterior */

    /**
     * Computes the posterior means of the heights u of X and v of Y by integrating the
     * likelihood given the root sequence on a grid over 0 < u < v < 1, with the density of
     * the heights being uniform.
     */
    private double[] computeExpectedHeights(Alignment alignment, int[] rootSequence) {
        int[][] tipStates = new int[TAXA.length][];
        for (int i = 0; i < TAXA.length; i++) {
            tipStates[i] = alignment.getCounts().get(alignment.getTaxonIndex(TAXA[i])).stream().mapToInt(Integer::intValue).toArray();
        }

        double step = ROOT_HEIGHT / NUM_GRID_POINTS;
        double totalWeight = 0.0;
        double sumU = 0.0;
        double sumV = 0.0;

        for (int i = 0; i < NUM_GRID_POINTS; i++) {
            for (int j = 0; j < NUM_GRID_POINTS; j++) {
                double u = (i + 0.5) * step;
                double v = (j + 0.5) * step;
                if (u >= v) {
                    continue;
                }

                double weight = this.computeLikelihood(tipStates, rootSequence, u, v);
                totalWeight += weight;
                sumU += weight * u;
                sumV += weight * v;
            }
        }

        return new double[]{sumU / totalWeight, sumV / totalWeight};
    }

    private double computeLikelihood(int[][] tipStates, int[] rootSequence, double u, double v) {
        double likelihood = 1.0;
        for (int site = 0; site < rootSequence.length; site++) {
            double[] cherry = this.multiply(
                    this.propagate(this.getTipPartials(tipStates, site, 0), u),
                    this.propagate(this.getTipPartials(tipStates, site, 1), u)
            );
            double[] inner = this.multiply(this.propagate(cherry, v - u), this.propagate(this.getTipPartials(tipStates, site, 2), v));
            double[] root = this.multiply(
                    this.propagate(inner, ROOT_HEIGHT - v),
                    this.propagate(this.getTipPartials(tipStates, site, 3), ROOT_HEIGHT)
            );
            likelihood *= root[rootSequence[site]];
        }
        return likelihood;
    }

    private double[] getTipPartials(int[][] tipStates, int site, int taxon) {
        double[] partials = new double[4];
        partials[tipStates[taxon][site]] = 1.0;
        return partials;
    }

    /** Returns the partials at the top of a branch of the given length from those at its bottom under Jukes-Cantor. */
    private double[] propagate(double[] partials, double length) {
        double stayProbability = 0.25 * (1.0 + 3.0 * Math.exp(-4.0 / 3.0 * length));
        double changeProbability = 0.25 * (1.0 - Math.exp(-4.0 / 3.0 * length));

        double[] result = new double[4];
        for (int from = 0; from < 4; from++) {
            for (int to = 0; to < 4; to++) {
                result[from] += (from == to ? stayProbability : changeProbability) * partials[to];
            }
        }
        return result;
    }

    private double[] multiply(double[] left, double[] right) {
        double[] result = new double[4];
        for (int state = 0; state < 4; state++) {
            result[state] = left[state] * right[state];
        }
        return result;
    }

    private Node getInternalChild(Node node) {
        return node.getLeft().isLeaf() ? node.getRight() : node.getLeft();
    }

}
