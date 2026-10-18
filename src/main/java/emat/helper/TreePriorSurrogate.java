package emat.helper;

import beast.base.core.BEASTObject;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.inference.StateNode;
import beast.base.spec.type.IntVector;
import beast.base.spec.type.RealScalar;
import beast.base.spec.type.RealVector;

import java.util.ArrayList;
import java.util.List;

@Description("A learned approximation of the log tree prior as a function of the height of a single node, which " +
        "several operators can share. Operators that know the change of the log tree prior of their moves, such as " +
        "the batch node displacement, teach it, and other operators draw node heights from it.")
public class TreePriorSurrogate extends BEASTObject {

    /** How the approximation of the tree prior depends on the midpoint of the range of a node. */
    public enum Shape {
        polynomial, spline, global
    }

    final public Input<Integer> degreeInput = new Input<>("degree", "the degree of the polynomial that approximates the log tree prior around the midpoint of the range of a node, if its shape is polynomial or spline", 3);
    final public Input<Shape> shapeInput = new Input<>("shape", "how the approximation of the tree prior depends on the midpoint of the range of a node: as a polynomial in the midpoint, as a polynomial in the offset whose coefficients are linear splines in the midpoint, or as a single linear spline of the slope along the absolute height", Shape.polynomial, Shape.values());
    final public Input<Integer> midpointDegreeInput = new Input<>("midpointDegree", "the highest power of the midpoint of the range of a node that the approximation of the tree prior may depend on, if its shape is polynomial", 3);
    final public Input<Integer> knotsInput = new Input<>("knots", "the number of knots along the height between which the approximation of the tree prior is interpolated, if its shape is spline or global", 20);
    final public Input<List<StateNode>> parametersInput = new Input<>("parameter", "real scalar, real vector or integer vector parameters of the tree prior, which the approximation of the tree prior may depend on", new ArrayList<>());
    final public Input<Integer> warmUpInput = new Input<>("warmUp", "the number of observations after which the approximation of the tree prior is fitted for the first time", 500);

    // the ridge penalty per observation of the fit of the approximation
    private static final double RIDGE_PENALTY = 1e-8;

    private LearnedTreePrior approximation;

    // the values of the parameters of the tree prior when they were last read
    private double[] parameterValues;

    @Override
    public void initAndValidate() {
        if (this.degreeInput.get() < 1) {
            throw new IllegalArgumentException("The degree of the tree prior approximation must be at least 1.");
        }

        this.parameterValues = new double[this.countParameterValues()];
        this.approximation = this.createApproximation();
    }

    /** Returns the number of values of the parameters of the tree prior, and checks that they are real or integer vectors. */
    private int countParameterValues() {
        int numValues = 0;
        for (StateNode parameter : this.parametersInput.get()) {
            if (parameter instanceof RealScalar<?>) {
                numValues++;
            } else if (parameter instanceof RealVector<?> vector) {
                numValues += vector.size();
            } else if (parameter instanceof IntVector<?> vector) {
                numValues += vector.size();
            } else {
                throw new IllegalArgumentException("The tree prior parameter " + parameter.getID() + " is neither a real scalar nor a real or integer vector.");
            }
        }
        return numValues;
    }

    private LearnedTreePrior createApproximation() {
        int degree = this.degreeInput.get();
        int numParameters = this.parameterValues.length;
        int warmUp = this.warmUpInput.get();

        return switch (this.shapeInput.get()) {
            case spline -> new LinearSplineTreePriorApproximation(degree, this.knotsInput.get(), numParameters, warmUp, RIDGE_PENALTY);
            case global -> new GlobalSplineTreePriorApproximation(this.knotsInput.get(), numParameters, warmUp, RIDGE_PENALTY);
            case polynomial -> new TreePriorApproximation(degree, this.midpointDegreeInput.get(), numParameters, warmUp, RIDGE_PENALTY);
        };
    }

    /** Returns the learned approximation, which draws heights and is taught by observed changes of the log tree prior. */
    public LearnedTreePrior getApproximation() {
        return this.approximation;
    }

    /**
     * Reads the current values of the parameters of the tree prior and returns them in the order the approximation
     * expects them. The array is reused by the next call.
     */
    public double[] readParameterValues() {
        int index = 0;
        for (StateNode parameter : this.parametersInput.get()) {
            if (parameter instanceof RealScalar<?> scalar) {
                this.parameterValues[index++] = scalar.get();
            } else if (parameter instanceof RealVector<?> vector) {
                for (int i = 0; i < vector.size(); i++) {
                    this.parameterValues[index++] = vector.get(i);
                }
            } else if (parameter instanceof IntVector<?> vector) {
                for (int i = 0; i < vector.size(); i++) {
                    this.parameterValues[index++] = vector.get(i);
                }
            }
        }
        return this.parameterValues;
    }

}
