package emat.operators.categories;

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
import emat.prior.GeneticPrior;
import emat.prior.SiteRateCategories;
import emat.state.Mutation;
import emat.state.Mutations;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class GammaShapeCategoryOperatorTest {

    private static final double CLOCK_RATE = 0.5;

    /** The objects of a genetic prior with three sites, two rate categories, a shape parameter and a fixed history. */
    private record Setup(Alignment alignment, TreeParser tree, SiteModel siteModel, RealScalarParam<PositiveReal> shape,
                         StrictClockModel clockModel, int[] categories, Mutations mutations) {
    }

    /**
     * Checks that the Hastings ratio of the shape operator is consistent with the posterior of the
     * shape with the categories summed out. For a proposal from the shape a with categories c to the
     * shape a' with categories c', the ratio of the posteriors of the joint states, times the Hastings
     * ratio, must equal the ratio of the posteriors with the categories summed out, times the ratio of
     * the proposed shapes. The joint posterior is the genetic prior of the state, times the proportion of
     * every category, which is one half here.
     */
    @Test
    public void testHastingsRatioMatchesCategorySummedPosterior() {
        Randomizer.setSeed(7);

        Setup setup = createSetup(new int[]{0, 1, 0});
        GeneticPrior geneticPrior = createGeneticPrior(setup);
        GammaShapeCategoryOperator operator = new GammaShapeCategoryOperator();
        operator.initByName("weight", 1.0, "mutations", setup.mutations, "geneticPrior", geneticPrior,
                "shape", setup.shape);

        int numTrials = 200;
        for (int trial = 0; trial < numTrials; trial++) {
            double oldShape = setup.shape.get();
            double oldLogSummedPosterior = computeLogSummedPosterior(setup);
            double oldLogJointPosterior = computeLogJointPosterior(setup);

            double logHastingsRatio = operator.proposal();
            if (logHastingsRatio == Double.NEGATIVE_INFINITY) {
                continue;
            }

            // the operator redraws the categories of its genetic prior, which the posteriors of the setup read
            SiteRateCategories categories = geneticPrior.getSiteRateCategories();
            for (int site = 0; site < setup.categories.length; site++) {
                setup.categories[site] = categories.getStateCategory(site);
            }

            double newShape = setup.shape.get();
            double newLogSummedPosterior = computeLogSummedPosterior(setup);
            double newLogJointPosterior = computeLogJointPosterior(setup);

            double expectedLogRatio = newLogSummedPosterior - oldLogSummedPosterior + Math.log(newShape / oldShape);
            double actualLogRatio = newLogJointPosterior - oldLogJointPosterior + logHastingsRatio;
            assertEquals(expectedLogRatio, actualLogRatio, 1e-8, "trial " + trial + ".");
        }
    }

    /**
     * Checks that a rejected proposal restores the categories of the state. BEAST stores the
     * calculation nodes only after the proposal, so the genetic prior is stored when the operator
     * has already redrawn the categories, and the restore still has to bring back the old ones.
     */
    @Test
    public void testRejectRestoresCategories() {
        Randomizer.setSeed(11);

        Setup setup = createSetup(new int[]{0, 1, 0});
        GeneticPrior geneticPrior = createGeneticPrior(setup);
        geneticPrior.calculateLogP();
        GammaShapeCategoryOperator operator = new GammaShapeCategoryOperator();
        operator.initByName("weight", 1.0, "mutations", setup.mutations, "geneticPrior", geneticPrior,
                "shape", setup.shape);

        SiteRateCategories categories = geneticPrior.getSiteRateCategories();
        int numSites = setup.categories.length;

        int numChangedTrials = 0;
        for (int trial = 0; trial < 200; trial++) {
            int[] oldCategories = new int[numSites];
            for (int site = 0; site < numSites; site++) {
                oldCategories[site] = categories.getStateCategory(site);
            }
            double oldShape = setup.shape.get();

            // the order of a rejected step of the MCMC: propose, store and calculate, then restore

            if (operator.proposal() == Double.NEGATIVE_INFINITY) {
                continue;
            }
            geneticPrior.store();
            geneticPrior.calculateLogP();
            geneticPrior.restore();
            setup.shape.set(oldShape);

            for (int site = 0; site < numSites; site++) {
                assertEquals(oldCategories[site], categories.getStateCategory(site), "state category of site " + site + " in trial " + trial + ".");
                assertEquals(oldCategories[site], categories.getCategory(site), "snapshot category of site " + site + " in trial " + trial + ".");
            }
            numChangedTrials++;
        }
        assertTrue(numChangedTrials > 0);
    }

    /**
     * Returns the log of the joint posterior of the current shape and categories, which is the
     * genetic prior of the state times the proportions of the categories of the sites. The genetic
     * prior is computed from scratch, so that no cached branch is reused after the shape changed.
     */
    private static double computeLogJointPosterior(Setup setup) {
        double[] categoryProportions = setup.siteModel.getCategoryProportions(setup.tree.getRoot());
        double logJointPosterior = createGeneticPrior(setup).calculateLogP();
        for (int site = 0; site < setup.categories.length; site++) {
            logJointPosterior += Math.log(categoryProportions[setup.categories[site]]);
        }
        return logJointPosterior;
    }

    /**
     * Returns the log of the posterior of the current shape with the categories of all sites
     * summed out, which enumerates every assignment of the categories of the three sites and then
     * restores the categories of the setup.
     */
    private static double computeLogSummedPosterior(Setup setup) {
        int numSites = setup.categories.length;
        int[] originalCategories = setup.categories.clone();

        double[] logJointPosteriors = new double[1 << numSites];
        for (int assignment = 0; assignment < logJointPosteriors.length; assignment++) {
            for (int site = 0; site < numSites; site++) {
                setup.categories[site] = (assignment >> site) & 1;
            }
            logJointPosteriors[assignment] = computeLogJointPosterior(setup);
        }

        System.arraycopy(originalCategories, 0, setup.categories, 0, numSites);

        double maxLogJointPosterior = Double.NEGATIVE_INFINITY;
        for (double logJointPosterior : logJointPosteriors) {
            maxLogJointPosterior = Math.max(maxLogJointPosterior, logJointPosterior);
        }
        double sum = 0.0;
        for (double logJointPosterior : logJointPosteriors) {
            sum += Math.exp(logJointPosterior - maxLogJointPosterior);
        }
        return maxLogJointPosterior + Math.log(sum);
    }

    /**
     * Creates a genetic prior on the objects of the setup with the categories of the setup, which
     * computes every branch from scratch on its first calculation.
     */
    private static GeneticPrior createGeneticPrior(Setup setup) {
        GeneticPrior geneticPrior = new GeneticPrior();
        geneticPrior.initByName(
                "data", setup.alignment,
                "tree", setup.tree,
                "siteModel", setup.siteModel,
                "branchRateModel", setup.clockModel,
                "mutations", setup.mutations
        );

        SiteRateCategories categories = geneticPrior.getSiteRateCategories();
        for (int site = 0; site < setup.categories.length; site++) {
            categories.setStateCategory(site, setup.categories[site]);
        }
        categories.markRedrawn();
        return geneticPrior;
    }

    /**
     * Creates the setup with a fixed history: one mutation on the branch above the tip B at site 1,
     * and one on the branch above the clade of C and D at site 0.
     */
    private static Setup createSetup(int[] categoryValues) {
        Alignment alignment = new Alignment();
        alignment.initByName(
                "sequence", List.of(
                        new Sequence("A", "ACA"),
                        new Sequence("B", "ATA"),
                        new Sequence("C", "GCA"),
                        new Sequence("D", "GCA")
                ),
                "dataType", "nucleotide"
        );

        TreeParser tree = new TreeParser();
        tree.initByName(
                "newick", "((A:1,B:1):1,(C:0.5,D:0.5):1.5);",
                "IsLabelledNewick", true,
                "adjustTipHeights", false
        );

        // the reference is the sequence of A, so B needs C -> T at site 1 and (C,D) needs A -> G at site 0

        Node nodeB = getTip(tree, "B");
        Node nodeCD = getTip(tree, "C").getParent();

        List<List<Mutation>> mutationsAboveNode = new ArrayList<>();
        for (int nodeNr = 0; nodeNr < tree.getNodeCount(); nodeNr++) {
            mutationsAboveNode.add(new ArrayList<>());
        }
        mutationsAboveNode.get(nodeB.getNr()).add(new Mutation(nodeB.getNr(), 0.4, Mutation.BRANCH_START, 1, 1, 3));
        mutationsAboveNode.get(nodeCD.getNr()).add(new Mutation(nodeCD.getNr(), 1.2, Mutation.BRANCH_START, 0, 0, 2));

        Mutations mutations = new Mutations();
        mutations.initByName("tree", tree, "alignment", alignment);
        mutations.initialiseMutations(mutationsAboveNode);

        RealScalarParam<PositiveReal> shape = new RealScalarParam<>(0.5, PositiveReal.INSTANCE);
        SiteModel siteModel = new SiteModel();
        siteModel.initByName("substModel", new JukesCantor(), "gammaCategoryCount", 2, "shape", shape);

        StrictClockModel clockModel = new StrictClockModel();
        clockModel.initByName("clock.rate", new RealScalarParam<>(CLOCK_RATE, PositiveReal.INSTANCE));

        return new Setup(alignment, tree, siteModel, shape, clockModel, categoryValues.clone(), mutations);
    }

    /** Returns the tip with the given taxon name. */
    private static Node getTip(TreeParser tree, String taxonName) {
        for (Node node : tree.getExternalNodes()) {
            if (node.getID().equals(taxonName)) {
                return node;
            }
        }
        throw new IllegalArgumentException("No tip " + taxonName + ".");
    }

}
