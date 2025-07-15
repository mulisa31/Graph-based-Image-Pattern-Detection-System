// Gr<Node> path;aph.java
package graph;

import java.util.*;

public class Graph {
    private final List<Node> nodes;
    private final List<Edge> edges;
    private final Map<Node, List<Edge>> adjacencyList;
    private List<Node> path;

    public Graph() {
        this.nodes = new ArrayList<>();
        this.edges = new ArrayList<>();
        this.adjacencyList = new HashMap<>();
    }
    
    //Copy constructor
    public Graph(Graph other) {
        this.nodes = new ArrayList<>(other.nodes);
        this.edges = new ArrayList<>(other.edges);
        this.path = other.path != null ? new ArrayList<>(other.path) : null;
        this.adjacencyList = new HashMap<>();
        other.adjacencyList.forEach((k, v) -> 
            this.adjacencyList.put(k, new ArrayList<>(v)));
        this.path = other.path != null ? new ArrayList<>(other.path) : null;
    }

    // Node operations
    public void addNode(Node node) {
        if (!nodes.contains(node)) {
            nodes.add(node);
            adjacencyList.put(node, new ArrayList<>());
        }
    }

    public List<Node> getNodes() {
        return Collections.unmodifiableList(nodes);
    }

    public void setPath(List<Node> path) {
        this.path = new ArrayList<>(path);
    }

    public List<Node> getPath() {
        return path != null ? Collections.unmodifiableList(path) : null;
    }
    
    // Edge operations
    public void addEdge(Node source, Node target, double weight) {
        if (nodes.contains(source) && nodes.contains(target)) {
            Edge edge = new Edge(source, target, weight);
            edges.add(edge);
            adjacencyList.get(source).add(edge);
            adjacencyList.get(target).add(new Edge(target, source, weight)); // Undirected graph
        }
    }

    public List<Edge> getEdges() {
        return Collections.unmodifiableList(edges);
    }

    public List<Edge> getEdgesFrom(Node node) {
        return Collections.unmodifiableList(adjacencyList.getOrDefault(node, new ArrayList<>()));
    }

    // Graph algorithms
    public List<Node> findShortestPath(Node start, Node end) {
        Map<Node, Double> distances = new HashMap<>();
        Map<Node, Node> previousNodes = new HashMap<>();
        PriorityQueue<Node> queue = new PriorityQueue<>(Comparator.comparingDouble(distances::get));

        // Initialize distances
        for (Node node : nodes) {
            distances.put(node, Double.MAX_VALUE);
        }
        distances.put(start, 0.0);
        queue.add(start);

        // Dijkstra's algorithm
        while (!queue.isEmpty()) {
            Node current = queue.poll();

            if (current.equals(end)) {
                break; // Found the shortest path
            }

            for (Edge edge : adjacencyList.get(current)) {
                Node neighbor = edge.getTarget();
                double newDistance = distances.get(current) + edge.getWeight();

                if (newDistance < distances.get(neighbor)) {
                    distances.put(neighbor, newDistance);
                    previousNodes.put(neighbor, current);
                    queue.add(neighbor);
                }
            }
        }

        // Reconstruct path
        List<Node> path = new ArrayList<>();
        for (Node at = end; at != null; at = previousNodes.get(at)) {
            path.add(at);
        }
        Collections.reverse(path);

        return path.size() > 1 ? path : Collections.emptyList();
    }

}