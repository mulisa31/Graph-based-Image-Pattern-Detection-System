package graph;

import java.util.*;

public class Region {
    public int id;
    public int centroidX, centroidY;
    public double meanColor;
    public List<Integer> neighborIds = new ArrayList<>();

    public Region(int id, int x, int y, double meanColor) {
        this.id = id;
        this.centroidX = x;
        this.centroidY = y;
        this.meanColor = meanColor;
    }

    public void addNeighbor(int neighborId) {
        if (!neighborIds.contains(neighborId)) {
            neighborIds.add(neighborId);
        }
    }

    public List<Integer> getNeighborIds() {
        return neighborIds;
    }
}

