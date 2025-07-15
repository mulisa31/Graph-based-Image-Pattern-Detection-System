//Node class

package graph;

import javafx.scene.paint.Color;

public class Node {
    private static int idCounter = 0;  // For generating unique IDs

    private final int id;          // Unique identifier
    private final double x, y;     // Coordinates (for image positioning)
    private  Color color;     // Color (for image processing)
    private int regionId = -1;     // For RAGs (-1 means unassigned)

    private double[] features;  // This will store features for KNN/MST

    // Constructors
    public Node(double x, double y, Color color) {
        this.id = idCounter++;
        this.x = x;
        this.y = y;
        this.color = color;
    }

    public Node(double x, double y) {
        this.id = idCounter++;
        this.x = x;
        this.y = y;
    }

    // Getters
    public int getId() { return id; }
    public double getX() { return x; }
    public double getY() { return y; }
    public Color getColor() { return color; }
    public int getRegionId() { return regionId; }

    // Setters
    public void setRegionId(int regionId) { this.regionId = regionId; }
    public void setColor(Color color) { this.color = color; }

    // Feature accessors
    public double[] getFeatures() {
        return features != null ? features : new double[] { x, y };
    }

    public void setFeatures(double[] features) {
        this.features = features;
    }

    // Equals and hashCode
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        Node other = (Node) obj;
        return id == other.id;
    }

    @Override
    public int hashCode() {
        return id;
    }

    // Debugging
    @Override
    public String toString() {
        return String.format(
            "Node(id=%d, x=%.1f, y=%.1f, region=%d, color=[%.2f,%.2f,%.2f])",
            id, x, y, regionId,
            color != null ? color.getRed() / 255.0 : 0.0,
            color != null ? color.getGreen() / 255.0 : 0.0,
            color != null ? color.getBlue() / 255.0 : 0.0
        );
    }
}
