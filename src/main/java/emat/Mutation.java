package emat;

/**
 * A single mutation at one site on the branch above the given node. Both times are BEAST
 * node heights, which grow backwards in time. Evolution runs forwards in time from the
 * parent to the node, so oldState is the state closer to the root and time is smaller than
 * timeOfPreviousMutation.
 */
public record Mutation(int nodeNr, double time, double timeOfPreviousMutation, int site, int oldState, int newState)  {

}
