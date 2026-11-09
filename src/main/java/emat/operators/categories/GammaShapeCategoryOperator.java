package emat.operators.categories;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.Operator;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.util.Randomizer;
import emat.prior.GeneticPrior;
import emat.prior.SiteRateCategories;
import emat.state.Mutations;
import emat.stochasticmapping.StochasticMapping;
import emat.stochasticmapping.SubstitutionProcess;

import java.text.DecimalFormat;

@Description("Proposes a new shape of the gamma site rates on a log scale, and redraws the category of every site from " +
        "its conditional distribution under the new shape, as described by SiteCategoryPosterior. The Hastings ratio " +
        "is the ratio of the conditional probabilities of the old and the new categories, which cancels their terms " +
        "in the posterior. The move is therefore accepted with the ratio of the posteriors of the shape with the " +
        "categories integrated out, which does not suffer from the coupling between the shape and the categories. " +
        "The half width of the proposal is tuned towards the target acceptance probability by the Robbins-Monro rule. " +
        "Requires a genetic prior with more than one rate category.")
public class GammaShapeCategoryOperator extends Operator {

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to operate on", Input.Validate.REQUIRED);
    final public Input<GeneticPrior> geneticPriorInput = new Input<>("geneticPrior", "the genetic prior that defines the evolutionary model and holds the categories", Input.Validate.REQUIRED);
    final public Input<RealScalarParam<PositiveReal>> shapeInput = new Input<>("shape", "the shape of the gamma site rates", Input.Validate.REQUIRED);
    final public Input<Double> scaleFactorInput = new Input<>("scaleFactor", "the half width of the uniform proposal of the log shape", 0.5);
    final public Input<Boolean> optimiseInput = new Input<>("optimise", "whether to tune the scale factor towards the target acceptance probability", true);
    final public Input<Double> targetAcceptanceInput = new Input<>("targetAcceptance", "the acceptance probability the scale factor is tuned towards", 0.44);

    private static final double MIN_SCALE_FACTOR = 1e-8;
    private static final double MAX_SCALE_FACTOR = 10.0;

    Mutations mutations;
    TreeInterface tree;
    GeneticPrior geneticPrior;
    RealScalarParam<PositiveReal> shape;
    SiteRateCategories siteRateCategories;
    SubstitutionProcess process;
    SiteCategoryPosterior posterior;
    double scaleFactor;

    int numSites;

    // reusable probabilities of the categories of a site, and the drawn categories of all sites
    double[] categoryProbabilities;
    int[] drawnCategories;

    @Override
    public void initAndValidate() {
        this.mutations = this.mutationsInput.get();
        this.tree = this.mutations.getTree();
        this.geneticPrior = this.geneticPriorInput.get();
        this.shape = this.shapeInput.get();
        if (this.geneticPrior.getNumCategories() < 2) {
            throw new IllegalArgumentException("The genetic prior has a single rate category, so there are no categories to redraw.");
        }
        this.siteRateCategories = this.geneticPrior.getSiteRateCategories();
        this.setCoercableParameterValue(this.scaleFactorInput.get());

        int numStates = this.mutations.getAlignment().getMaxStateCount();
        this.process = new SubstitutionProcess(this.geneticPrior, this.tree, numStates);
        this.posterior = new SiteCategoryPosterior(this.mutations, this.tree, this.process);

        this.numSites = this.mutations.getReferenceSequence().length;
        this.categoryProbabilities = new double[this.posterior.getNumCategories()];
        this.drawnCategories = new int[this.numSites];
    }

    /**
     * Proposes the new shape, then draws the categories of all sites under it. The Hastings ratio
     * is the log proposal ratio of the shape plus the log probability of the old categories under
     * the old shape, minus the log probability of the new categories under the new shape.
     */
    @Override
    public double proposal() {
        double oldShape = this.shape.get();
        double factor = Math.exp(this.scaleFactor * (2.0 * Randomizer.nextDouble() - 1.0));
        double newShape = oldShape * factor;
        if (newShape < this.shape.getLower() || newShape > this.shape.getUpper()) {
            return Double.NEGATIVE_INFINITY;
        }

        // collect the statistics of the sites, which do not depend on the shape, once for both shapes

        this.process.update();
        this.posterior.collectSiteStatistics();

        double logOldCategoryProbability = this.computeLogCurrentCategoryProbability();

        this.shape.set(newShape);
        double logNewCategoryProbability = this.drawCategories();

        // an extreme shape can make the category probabilities degenerate, so such a proposal is
        // rejected before the categories of the state change, as a failed proposal is not restored
        double logHastingsRatio = Math.log(factor) + logOldCategoryProbability - logNewCategoryProbability;
        if (Double.isNaN(logHastingsRatio)) {
            return Double.NEGATIVE_INFINITY;
        }

        this.setDrawnCategories();
        return logHastingsRatio;
    }

    /**
     * Returns the log probability of the current categories of all sites given the histories and
     * the current shape.
     */
    private double computeLogCurrentCategoryProbability() {
        this.updateProportions();

        double logProbability = 0.0;
        for (int site = 0; site < this.numSites; site++) {
            logProbability += this.posterior.computeLogProbability(site, this.siteRateCategories.getStateCategory(site));
        }
        return logProbability;
    }

    /**
     * Draws the category of every site from its conditional distribution under the current shape
     * into the drawn categories, and returns the log probability of the drawn categories.
     */
    private double drawCategories() {
        this.updateProportions();

        double logProbability = 0.0;
        for (int site = 0; site < this.numSites; site++) {
            this.posterior.computeProbabilities(site, this.categoryProbabilities);
            int category = StochasticMapping.sampleIndex(this.categoryProbabilities);
            this.drawnCategories[site] = category;
            logProbability += this.posterior.computeLogProbability(site, category);
        }
        return logProbability;
    }

    /** Sets the drawn categories as the categories of the state. */
    private void setDrawnCategories() {
        for (int site = 0; site < this.numSites; site++) {
            this.siteRateCategories.setStateCategory(site, this.drawnCategories[site]);
        }

        // the categories of all sites are drawn together, so they are marked as redrawn as a whole
        this.siteRateCategories.markRedrawn();
    }

    /** Updates the rates of the process and the category proportions of the posterior under the current shape. */
    private void updateProportions() {
        this.process.update();
        this.posterior.setCategoryProportions(this.geneticPrior.siteModel.getCategoryProportions(this.tree.getRoot()));
    }

    /* Tuning */

    /**
     * Tunes the log of the scale factor by the Robbins-Monro rule towards the target acceptance
     * probability. An undefined acceptance ratio carries no information about the step size, so it
     * is skipped.
     */
    @Override
    public void optimize(double logAlpha) {
        if (this.optimiseInput.get() && !Double.isNaN(logAlpha)) {
            double delta = this.calcDelta(logAlpha) + Math.log(this.getCoercableParameterValue());
            this.setCoercableParameterValue(Math.exp(delta));
        }
    }

    @Override
    public double getTargetAcceptanceProbability() {
        return this.targetAcceptanceInput.get();
    }

    @Override
    public double getCoercableParameterValue() {
        return this.scaleFactor;
    }

    @Override
    public void setCoercableParameterValue(double value) {
        this.scaleFactor = Math.max(Math.min(value, MAX_SCALE_FACTOR), MIN_SCALE_FACTOR);
    }

    @Override
    public String getPerformanceSuggestion() {
        double acceptanceProbability = this.m_nNrAccepted / (this.m_nNrAccepted + this.m_nNrRejected + 0.0);
        double ratio = acceptanceProbability / this.getTargetAcceptanceProbability();
        ratio = Math.max(Math.min(ratio, 2.0), 0.5);

        DecimalFormat formatter = new DecimalFormat("#.###");
        if (ratio < 0.75 || ratio > 1.25) {
            return "Try setting scale factor to about " + formatter.format(this.getCoercableParameterValue() * ratio);
        }
        return "";
    }

}
