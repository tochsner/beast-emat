package emat.helper;

import beast.base.core.BEASTObject;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.core.Log;
import beast.base.core.Loggable;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.StateNode;
import beast.base.internal.json.JSONArray;
import beast.base.internal.json.JSONException;
import beast.base.internal.json.JSONObject;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.spec.inference.parameter.RealVectorParam;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Description("Lets parameters of the tree prior follow the tree: when the log tree length changes by Δ, every " +
        "parameter is shifted by its coefficient times Δ, on the log scale for a parameter that cannot be negative " +
        "and on its own scale otherwise. The coefficients are the regression slopes of the parameters on the log " +
        "tree length, learned from the chain (docs/parameter-following-operator.md).")
public class ParameterFollowing extends BEASTObject implements Loggable {

    final public Input<Tree> treeInput = new Input<>("tree", "the tree whose length the parameters follow", Input.Validate.REQUIRED);
    final public Input<List<StateNode>> parametersInput = new Input<>("parameter", "real scalar or vector parameters that follow the tree", new ArrayList<>());
    final public Input<Integer> warmUpInput = new Input<>("warmUp", "the number of observations after which the coefficients are fitted for the first time", 2000);
    final public Input<Boolean> optimiseInput = new Input<>("optimise", "whether the coefficients are learned from the chain", true);
    final public Input<List<Double>> coefficientsInput = new Input<>("coefficient", "the coefficients per parameter dimension to start from, which stay fixed if the coefficients are not learned", new ArrayList<>());

    // the factor by which the number of observations grows between two fits
    private static final double REFIT_GROWTH = 1.5;

    /** One dimension of a followed parameter. The index is -1 for a scalar. */
    private record Dimension(StateNode parameter, int index, boolean isLogScale) {
    }

    private Tree tree;
    private List<Dimension> dimensions;
    private double[] coefficients;
    private boolean isFitted;

    // the first observation, which all sums are taken relative to
    private double referenceSummary;
    private double[] referenceValues;

    // running sums over all observations: Σs, Σs², then Σp_j and Σs·p_j per dimension
    private long numObservations;
    private double[] sums;
    private long nextFit;

    // the number of observations and the sums at every fit, to fit on the more recent part of the observations
    private final List<Long> snapshotCounts = new ArrayList<>();
    private final List<double[]> snapshotSums = new ArrayList<>();

    // reusable buffer for the values of the parameters on their scales
    private double[] values;

    @Override
    public void initAndValidate() {
        this.tree = this.treeInput.get();

        this.dimensions = new ArrayList<>();
        for (StateNode parameter : this.parametersInput.get()) {
            if (parameter instanceof RealScalarParam<?> scalar) {
                this.dimensions.add(new Dimension(parameter, -1, scalar.getLower() >= 0.0));
            } else if (parameter instanceof RealVectorParam<?> vector) {
                for (int i = 0; i < vector.size(); i++) {
                    this.dimensions.add(new Dimension(parameter, i, vector.getLower() >= 0.0));
                }
            } else {
                throw new IllegalArgumentException("The followed parameter " + parameter.getID() + " is neither a real scalar nor a real vector parameter.");
            }
        }
        if (this.dimensions.isEmpty()) {
            throw new IllegalArgumentException("At least one parameter must follow the tree.");
        }

        int numDimensions = this.dimensions.size();
        this.coefficients = new double[numDimensions];
        this.referenceValues = new double[numDimensions];
        this.values = new double[numDimensions];
        this.sums = new double[2 + 2 * numDimensions];

        List<Double> initialCoefficients = this.coefficientsInput.get();
        if (!initialCoefficients.isEmpty()) {
            if (initialCoefficients.size() != numDimensions) {
                throw new IllegalArgumentException("There must be one coefficient per dimension of the followed parameters (" + numDimensions + ").");
            }
            for (int j = 0; j < numDimensions; j++) {
                this.coefficients[j] = initialCoefficients.get(j);
            }
        } else if (!this.optimiseInput.get()) {
            throw new IllegalArgumentException("Coefficients must be given if they are not learned.");
        }

        this.nextFit = Math.max(this.warmUpInput.get(), 2);
    }

    public List<StateNode> getParameters() {
        return this.parametersInput.get();
    }

    public double[] getCoefficients() {
        return this.coefficients.clone();
    }

    public boolean isFitted() {
        return this.isFitted;
    }

    public long getNumObservations() {
        return this.numObservations;
    }

    /** Returns the summary of the tree that the parameters follow, which is the log of its length. */
    public double computeSummary() {
        double length = 0.0;
        for (Node node : this.tree.getNodesAsArray()) {
            if (!node.isRoot()) {
                length += node.getLength();
            }
        }
        return Math.log(length);
    }

    /**
     * Shifts every parameter by its coefficient times the given change of the summary, on its scale. Returns the
     * log of the Jacobian of the shift, which is the sum of the shifts on the log scale. If a parameter would
     * leave its domain, no parameter is changed and negative infinity is returned.
     */
    public double shiftParameters(double summaryChange) {
        double logJacobian = 0.0;

        for (int j = 0; j < this.dimensions.size(); j++) {
            Dimension dimension = this.dimensions.get(j);
            double shift = this.coefficients[j] * summaryChange;

            double newValue;
            if (dimension.isLogScale()) {
                newValue = this.getValue(dimension) * Math.exp(shift);
                logJacobian += shift;
            } else {
                newValue = this.getValue(dimension) + shift;
            }

            if (!Double.isFinite(newValue) || !this.isValid(dimension, newValue)) {
                return Double.NEGATIVE_INFINITY;
            }
            this.values[j] = newValue;
        }

        for (int j = 0; j < this.dimensions.size(); j++) {
            if (this.coefficients[j] != 0.0) {
                this.setValue(this.dimensions.get(j), this.values[j]);
            }
        }

        return logJacobian;
    }

    /* Learning */

    /**
     * Records the given summary of the current tree together with the current parameters, and refits the
     * coefficients if the number of observations has grown enough since the last fit.
     */
    public void observe(double summary) {
        if (!this.optimiseInput.get()) {
            return;
        }

        for (int j = 0; j < this.dimensions.size(); j++) {
            this.values[j] = this.getScaledValue(this.dimensions.get(j));
        }

        if (this.numObservations == 0) {
            this.referenceSummary = summary;
            System.arraycopy(this.values, 0, this.referenceValues, 0, this.values.length);
        }

        int numDimensions = this.dimensions.size();
        double s = summary - this.referenceSummary;
        this.sums[0] += s;
        this.sums[1] += s * s;
        for (int j = 0; j < numDimensions; j++) {
            double p = this.values[j] - this.referenceValues[j];
            this.sums[2 + j] += p;
            this.sums[2 + numDimensions + j] += s * p;
        }
        this.numObservations++;

        if (this.numObservations >= this.nextFit) {
            this.fit();
        }
    }

    /**
     * Sets every coefficient to the regression slope of its parameter on the summary, cov(p, s) / var(s), over
     * the observations since the last fit that was made with at most half of the current observations. Leaving
     * out the early observations keeps a drift at the start of the chain from setting the coefficients.
     */
    private void fit() {
        int numDimensions = this.dimensions.size();

        long startCount = 0;
        double[] startSums = new double[this.sums.length];
        for (int i = this.snapshotCounts.size() - 1; i >= 0; i--) {
            if (2 * this.snapshotCounts.get(i) <= this.numObservations) {
                startCount = this.snapshotCounts.get(i);
                startSums = this.snapshotSums.get(i);
                break;
            }
        }

        double count = this.numObservations - startCount;
        double meanSummary = (this.sums[0] - startSums[0]) / count;
        double summaryVariance = (this.sums[1] - startSums[1]) / count - meanSummary * meanSummary;

        if (summaryVariance > 1e-14) {
            boolean isFinite = true;
            double[] slopes = new double[numDimensions];
            for (int j = 0; j < numDimensions; j++) {
                double meanValue = (this.sums[2 + j] - startSums[2 + j]) / count;
                double covariance = (this.sums[2 + numDimensions + j] - startSums[2 + numDimensions + j]) / count - meanSummary * meanValue;
                slopes[j] = covariance / summaryVariance;
                isFinite &= Double.isFinite(slopes[j]);
            }

            if (isFinite) {
                this.coefficients = slopes;
                this.isFitted = true;
            }
        }

        if (this.snapshotCounts.isEmpty()) {
            Log.info.println("Parameter following " + this.getID() + ": warm-up reached after " + this.numObservations +
                    " observations, coefficients " + Arrays.toString(this.coefficients));
        }

        this.snapshotCounts.add(this.numObservations);
        this.snapshotSums.add(this.sums.clone());
        this.nextFit = (long) Math.ceil(this.numObservations * REFIT_GROWTH);
    }

    /* Parameters */

    private double getValue(Dimension dimension) {
        if (dimension.index() < 0) {
            return ((RealScalarParam<?>) dimension.parameter()).get();
        }
        return ((RealVectorParam<?>) dimension.parameter()).get(dimension.index());
    }

    private double getScaledValue(Dimension dimension) {
        double value = this.getValue(dimension);
        return dimension.isLogScale() ? Math.log(value) : value;
    }

    private boolean isValid(Dimension dimension, double value) {
        if (dimension.index() < 0) {
            return ((RealScalarParam<?>) dimension.parameter()).isValid(value);
        }
        return ((RealVectorParam<?>) dimension.parameter()).isValid(value);
    }

    private void setValue(Dimension dimension, double value) {
        if (dimension.index() < 0) {
            ((RealScalarParam<?>) dimension.parameter()).set(value);
        } else {
            ((RealVectorParam<?>) dimension.parameter()).set(dimension.index(), value);
        }
    }

    /* State File */

    /** Returns what has been learned so far, to be written to the state file. */
    public JSONObject storeToJson() throws JSONException {
        JSONObject json = new JSONObject();
        json.put("coefficients", this.toJsonArray(this.coefficients));
        json.put("isFitted", this.isFitted);
        json.put("numObservations", this.numObservations);
        json.put("nextFit", this.nextFit);
        json.put("referenceSummary", this.referenceSummary);
        json.put("referenceValues", this.toJsonArray(this.referenceValues));
        json.put("sums", this.toJsonArray(this.sums));

        JSONArray snapshots = new JSONArray();
        for (int i = 0; i < this.snapshotCounts.size(); i++) {
            JSONObject snapshot = new JSONObject();
            snapshot.put("numObservations", this.snapshotCounts.get(i).longValue());
            snapshot.put("sums", this.toJsonArray(this.snapshotSums.get(i)));
            snapshots.put(snapshot);
        }
        json.put("snapshots", snapshots);

        return json;
    }

    /** Restores what was written by {@link #storeToJson()}. A state for other parameters is ignored. */
    public void restoreFromJson(JSONObject json) throws JSONException {
        double[] storedCoefficients = this.toArray(json.getJSONArray("coefficients"));
        if (storedCoefficients.length != this.coefficients.length) {
            return;
        }

        this.coefficients = storedCoefficients;
        this.isFitted = json.getBoolean("isFitted");
        this.numObservations = json.getLong("numObservations");
        this.nextFit = json.getLong("nextFit");
        this.referenceSummary = json.getDouble("referenceSummary");
        this.referenceValues = this.toArray(json.getJSONArray("referenceValues"));
        this.sums = this.toArray(json.getJSONArray("sums"));

        this.snapshotCounts.clear();
        this.snapshotSums.clear();
        JSONArray snapshots = json.getJSONArray("snapshots");
        for (int i = 0; i < snapshots.length(); i++) {
            JSONObject snapshot = (JSONObject) snapshots.get(i);
            this.snapshotCounts.add(snapshot.getLong("numObservations"));
            this.snapshotSums.add(this.toArray(snapshot.getJSONArray("sums")));
        }
    }

    private JSONArray toJsonArray(double[] values) throws JSONException {
        JSONArray array = new JSONArray();
        for (double value : values) {
            array.put(value);
        }
        return array;
    }

    private double[] toArray(JSONArray array) throws JSONException {
        double[] values = new double[array.length()];
        for (int i = 0; i < values.length; i++) {
            values[i] = array.getDouble(i);
        }
        return values;
    }

    /* Logging */

    /**
     * Logs per dimension the coefficient and the part of the parameter that does not follow the tree, which is
     * the parameter on its scale minus the coefficient times the summary.
     */
    @Override
    public void init(PrintStream out) {
        String id = this.getID() == null ? "following" : this.getID();
        for (int j = 0; j < this.dimensions.size(); j++) {
            out.print(id + ".coefficient." + (j + 1) + "\t");
        }
        for (int j = 0; j < this.dimensions.size(); j++) {
            out.print(id + ".remainder." + (j + 1) + "\t");
        }
    }

    @Override
    public void log(long sample, PrintStream out) {
        double summary = this.computeSummary();
        for (int j = 0; j < this.dimensions.size(); j++) {
            out.print(this.coefficients[j] + "\t");
        }
        for (int j = 0; j < this.dimensions.size(); j++) {
            out.print((this.getScaledValue(this.dimensions.get(j)) - this.coefficients[j] * summary) + "\t");
        }
    }

    @Override
    public void close(PrintStream out) {
    }

}
