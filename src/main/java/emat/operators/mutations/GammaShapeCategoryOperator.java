package emat.operators.mutations;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.Operator;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.inference.parameter.IntVectorParam;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.util.Randomizer;
import emat.prior.GeneticPrior;
import emat.state.Mutations;
import emat.stochasticmapping.StochasticMapping;
import emat.stochasticmapping.SubstitutionProcess;

@Description("Proposes a new shape of the gamma site rates on a log scale, and redraws the category of every site from " +
        "its conditional distribution under the new shape, as described by SiteCategoryPosterior. The Hastings ratio " +
        "is the ratio of the conditional probabilities of the old and the new categories, which cancels their terms " +
        "in the posterior. The move is therefore accepted with the ratio of the posteriors of the shape with the " +
        "categories integrated out, which does not suffer from the coupling between the shape and the categories. " +
        "Requires a genetic prior with siteRateCategories.")
public class GammaShapeCategoryOperator extends Operator {

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to operate on", Input.Validate.REQUIRED);
    final public Input<GeneticPrior> geneticPriorInput = new Input<>("geneticPrior", "the genetic prior that defines the evolutionary model and holds the categories", Input.Validate.REQUIRED);
    final public Input<RealScalarParam<PositiveReal>> shapeInput = new Input<>("shape", "the shape of the gamma site rates", Input.Validate.REQUIRED);
    final public Input<Double> scaleFactorInput = new Input<>("scaleFactor", "the half width of the uniform proposal of the log shape", 0.5);

    Mutations mutations;
    TreeInterface tree;
    GeneticPrior geneticPrior;
    RealScalarParam<PositiveReal> shape;
    IntVectorParam<?> siteRateCategories;
    SubstitutionProcess process;
    SiteCategoryPosterior posterior;
    double scaleFactor;

    int numSites;

    // reusable probabilities of the categories of a site
    double[] categoryProbabilities;

    @Override
    public void initAndValidate() {
        this.mutations = this.mutationsInput.get();
        this.tree = this.mutations.getTree();
        this.geneticPrior = this.geneticPriorInput.get();
        this.shape = this.shapeInput.get();
        this.siteRateCategories = this.geneticPrior.getSiteRateCategories();
        if (this.siteRateCategories == null) {
            throw new IllegalArgumentException("The genetic prior has no siteRateCategories to redraw.");
        }
        this.scaleFactor = this.scaleFactorInput.get();

        int numStates = this.mutations.getAlignment().getMaxStateCount();
        this.process = new SubstitutionProcess(this.geneticPrior, this.tree, numStates);
        this.posterior = new SiteCategoryPosterior(this.mutations, this.tree, this.process);

        this.numSites = this.mutations.getReferenceSequence().length;
        this.categoryProbabilities = new double[this.posterior.getNumCategories()];
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

        double logOldCategoryProbability = this.computeLogCurrentCategoryProbability();

        this.shape.set(newShape);
        double logNewCategoryProbability = this.drawCategories();

        return Math.log(factor) + logOldCategoryProbability - logNewCategoryProbability;
    }

    /**
     * Returns the log probability of the current categories of all sites given the histories and
     * the current shape.
     */
    private double computeLogCurrentCategoryProbability() {
        this.updatePosterior();

        double logProbability = 0.0;
        for (int site = 0; site < this.numSites; site++) {
            logProbability += this.posterior.computeLogProbability(site, this.siteRateCategories.get(site));
        }
        return logProbability;
    }

    /**
     * Draws the category of every site from its conditional distribution under the current shape,
     * and returns the log probability of the drawn categories.
     */
    private double drawCategories() {
        this.updatePosterior();

        double logProbability = 0.0;
        boolean isChanged = false;
        for (int site = 0; site < this.numSites; site++) {
            this.posterior.computeProbabilities(site, this.categoryProbabilities);
            int category = StochasticMapping.sampleIndex(this.categoryProbabilities);
            if (category != this.siteRateCategories.get(site)) {
                this.siteRateCategories.set(site, category);
                isChanged = true;
            }
            logProbability += this.posterior.computeLogProbability(site, category);
        }
        if (isChanged) {
            this.geneticPrior.markSiteCategoriesChanged();
        }
        return logProbability;
    }

    /** Updates the rates of the process and the posterior of the sites under the current shape. */
    private void updatePosterior() {
        this.process.update();
        this.posterior.update(this.geneticPrior.siteModel.getCategoryProportions(this.tree.getRoot()));
    }

}
