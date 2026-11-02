package emat.operators.nodetimes.batch;

import adapters.Adapter;
import beast.base.inference.StateNode;
import beast.base.spec.type.IntVector;
import beast.base.spec.type.RealScalar;
import beast.base.spec.type.RealVector;

import java.util.List;

/**
 * The current values of the parameters of the tree prior, which the approximation of the tree prior may depend on.
 * The values of the parameters come first, followed by the mutable and the immutable values of the adapters.
 */
public class TreePriorParameterValues {

    // the adapters have no use for the node, as the parameters of the tree prior do not belong to one
    private static final int NO_NODE = -1;

    private final List<StateNode> parameters;
    private final List<Adapter> adapters;
    private final double[] values;

    /** Checks that the parameters are real scalars or real or integer vectors, and counts their values. */
    public TreePriorParameterValues(List<StateNode> parameters, List<Adapter> adapters) {
        this.parameters = parameters;
        this.adapters = adapters;
        this.values = new double[this.countValues()];
    }

    /** Returns the array that {@link #read} fills, which the approximation of the tree prior takes as it is. */
    public double[] getValues() {
        return this.values;
    }

    /** Reads the current values of the parameters, and then the mutable and immutable values of the adapters. */
    public void read() {
        int index = 0;
        for (StateNode parameter : this.parameters) {
            if (parameter instanceof RealScalar<?> scalar) {
                this.values[index++] = scalar.get();
            } else if (parameter instanceof RealVector<?> vector) {
                for (int i = 0; i < vector.size(); i++) {
                    this.values[index++] = vector.get(i);
                }
            } else if (parameter instanceof IntVector<?> vector) {
                for (int i = 0; i < vector.size(); i++) {
                    this.values[index++] = vector.get(i);
                }
            }
        }

        for (Adapter adapter : this.adapters) {
            for (double value : adapter.getMutable(NO_NODE)) {
                this.values[index++] = value;
            }
            for (double value : adapter.getImmutable(NO_NODE)) {
                this.values[index++] = value;
            }
        }
    }

    private int countValues() {
        int numValues = 0;
        for (Adapter adapter : this.adapters) {
            numValues += adapter.getNumMutable() + adapter.getNumImmutable();
        }
        for (StateNode parameter : this.parameters) {
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

}
