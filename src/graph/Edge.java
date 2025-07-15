//Edge class
package graph;

public class Edge {
    private final Node source;
    private final Node target;
    private final double weight;

    public Edge(Node source, Node target, double weight) {
        this.source = source;
        this.target = target;
        this.weight = weight;
    }

    // Getters
    public Node getSource() { return source; }
    public Node getTarget() { return target; }
    public double getWeight() { return weight; }
}
