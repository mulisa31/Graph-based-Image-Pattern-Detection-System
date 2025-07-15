package Implementations;

import javafx.application.Platform;
import javafx.scene.image.*;
import javafx.scene.layout.Pane;
import javafx.scene.paint.Color;
import javafx.scene.shape.*;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import javafx.stage.Window;

import java.io.File;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;
import java.util.function.DoubleConsumer;
import java.util.stream.Collectors;

import graph.Edge;
import graph.Graph;
import graph.Node;
import graph.Region;

public class AppLogic {
    private final ExecutorService executor = Executors.newFixedThreadPool(2);
    private Image currentImage;
    private Graph currentGraph;
    private Window parentWindow;
    private final Map<String, List<GraphFeatureVector>> trainingData = new HashMap<>();

    // Graph feature vector container
    private static class GraphFeatureVector {
        final double[] features;
        final String label;

        GraphFeatureVector(double[] features, String label) {
            this.features = features;
            this.label = label;
        }
    }

    public void setParentWindow(Window parentWindow) {
        this.parentWindow = parentWindow;
    }

    // Image Processing Methods
    public void handleOpenImage(Consumer<Image> imageConsumer, Consumer<String> errorHandler) {
        Platform.runLater(() -> {
            FileChooser fileChooser = new FileChooser();
            fileChooser.setTitle("Open Image File");
            fileChooser.getExtensionFilters().addAll(
                new FileChooser.ExtensionFilter("Image Files", "*.png", "*.jpg", "*.jpeg")
            );
            
            File file = fileChooser.showOpenDialog(parentWindow);
            if (file != null) {
                executor.execute(() -> {
                    try {
                        Image image = new Image(file.toURI().toString(),true);
                        image.progressProperty().addListener((obs, oldVal, progress) -> {
                            if (progress.doubleValue() == 1.0) {
                                currentImage = image;
                                Platform.runLater(() -> imageConsumer.accept(image));
                            }
                        });
                    } catch (Exception e) {
                        Platform.runLater(() -> errorHandler.accept("Error loading image: " + e.getMessage()));
                    }
                });
            }
        });
    }

    // Graph Construction Methods
    public void handleBuildCompareGraph(DoubleConsumer progressUpdater,
                               Consumer<Pane> graphConsumer,
                               Consumer<String> errorHandler) {
        if (currentImage == null) {
            errorHandler.accept("No image loaded. Please load an image first.");
            return;
        }
        
        executor.execute(() -> {
            try {
                Platform.runLater(() -> progressUpdater.accept(0.1));
                
                currentGraph = createRAGFromImage(currentImage);
                
                Platform.runLater(() -> progressUpdater.accept(0.8));
                
                Pane graphPane = visualizeGraph(currentGraph);
                
                Platform.runLater(() -> {
                    progressUpdater.accept(1.0);
                    graphConsumer.accept(graphPane);
                });
            } catch (Exception e) {
                Platform.runLater(() -> errorHandler.accept("Error building graph: " + e.getMessage()));
            }
        });
    }
    
    
    public void handleBuildGraph(DoubleConsumer progressUpdater,
		            Consumer<Graph> graphConsumer, 
		            Consumer<String> errorHandler) {
		if (currentImage == null) {
		errorHandler.accept("No image loaded. Please load an image first.");
		return;
		}
		
		executor.execute(() -> {
			try {
				Platform.runLater(() -> progressUpdater.accept(0.1));
				
				currentGraph = createRAGFromImage(currentImage);
				
				Platform.runLater(() -> progressUpdater.accept(0.8));
	
				Platform.runLater(() -> {
				 progressUpdater.accept(1.0);
				 graphConsumer.accept(currentGraph); 
				});
			} catch (Exception e) {
			Platform.runLater(() -> errorHandler.accept("Error building graph: " + e.getMessage()));
			}
		});
    }

    private Graph createRAGFromImage(Image image) {
        Graph graph = new Graph();
        PixelReader reader = image.getPixelReader();
        int width = (int) image.getWidth();
        int height = (int) image.getHeight();

        int superpixelSize = Math.max(10, Math.min(width, height) / 20);
        Map<Integer, Node> regionToNode = new HashMap<>();
        Map<Integer, Region> regionMap = new HashMap<>();

        int regionsPerRow = (width + superpixelSize - 1) / superpixelSize;

        for (int y = 0; y < height; y += superpixelSize) {
            for (int x = 0; x < width; x += superpixelSize) {
                int w = Math.min(superpixelSize, width - x);
                int h = Math.min(superpixelSize, height - y);
                Color avgColor = calculateAverageColor(reader, x, y, w, h);

                int regionId = (y / superpixelSize) * regionsPerRow + (x / superpixelSize);
                double centroidX = x + w / 2.0;
                double centroidY = y + h / 2.0;

                Node node = new Node(centroidX, centroidY, avgColor);
                node.setRegionId(regionId);
                graph.addNode(node);
                regionToNode.put(regionId, node);

                double grayMean = (avgColor.getRed() + avgColor.getGreen() + avgColor.getBlue()) / 3.0;
                Region region = new Region(regionId, (int) centroidX, (int) centroidY, grayMean);
                regionMap.put(regionId, region);

                // Extract features and assign to node
                double[] features = RegionFeatures.extract(region);
                node.setFeatures(features);
            }
        }

        for (int regionId : regionToNode.keySet()) {
            // Right neighbor (ensure not wrapping around)
            if ((regionId + 1) % regionsPerRow != 0 && regionToNode.containsKey(regionId + 1)) {
                createEdgeBetweenRegions(graph, regionToNode.get(regionId), regionToNode.get(regionId + 1));
            }
            // Bottom neighbor
            if (regionToNode.containsKey(regionId + regionsPerRow)) {
                createEdgeBetweenRegions(graph, regionToNode.get(regionId), regionToNode.get(regionId + regionsPerRow));
            }
        }

        return graph;
    }
    
    /*
    private Graph createRAGFromImage(Image image) {
        Graph graph = new Graph();
        PixelReader reader = image.getPixelReader();
        int width = (int) image.getWidth();
        int height = (int) image.getHeight();

        int superpixelSize = Math.max(10, Math.min(width, height) / 20);
        Map<Integer, Node> regionToNode = new HashMap<>();
        Map<Integer, Region> regionMap = new HashMap<>();

        int regionsPerRow = (width + superpixelSize - 1) / superpixelSize;

        for (int y = 0; y < height; y += superpixelSize) {
            for (int x = 0; x < width; x += superpixelSize) {
                int w = Math.min(superpixelSize, width - x);
                int h = Math.min(superpixelSize, height - y);
                Color avgColor = calculateAverageColor(reader, x, y, w, h);

                int regionId = (y / superpixelSize) * regionsPerRow + (x / superpixelSize);
                double centroidX = x + w / 2.0;
                double centroidY = y + h / 2.0;

                Node node = new Node(centroidX, centroidY, avgColor);
                node.setRegionId(regionId);
                graph.addNode(node);
                regionToNode.put(regionId, node);

                double grayMean = (avgColor.getRed() + avgColor.getGreen() + avgColor.getBlue()) / 3.0;
                Region region = new Region(regionId, (int) centroidX, (int) centroidY, grayMean);
                regionMap.put(regionId, region);

                // Extract features and assign to node
                double[] features = RegionFeature.extract(region);
                node.setFeatures(features);
            }
        }

        for (int regionId : regionToNode.keySet()) {
            // Right neighbor (ensure not wrapping around)
            if ((regionId + 1) % regionsPerRow != 0 && regionToNode.containsKey(regionId + 1)) {
                createEdgeBetweenRegions(graph, regionToNode.get(regionId), regionToNode.get(regionId + 1));
            }
            // Bottom neighbor
            if (regionToNode.containsKey(regionId + regionsPerRow)) {
                createEdgeBetweenRegions(graph, regionToNode.get(regionId), regionToNode.get(regionId + regionsPerRow));
            }
        }

        return graph;
    }
*/

    private Color calculateAverageColor(PixelReader reader, int startX, int startY, int width, int height) {
        double totalR = 0, totalG = 0, totalB = 0;
        int pixelCount = 0;

        for (int y = startY; y < startY + height; y++) {
            for (int x = startX; x < startX + width; x++) {
                Color color = reader.getColor(x, y);
                totalR += color.getRed();
                totalG += color.getGreen();
                totalB += color.getBlue();
                pixelCount++;
            }
        }

        return new Color(
            totalR / pixelCount,
            totalG / pixelCount,
            totalB / pixelCount,
            1.0
        );
    }

    private void createEdgeBetweenRegions(Graph graph, Node node1, Node node2) {
        double[] features1 = node1.getFeatures();
        double[] features2 = node2.getFeatures();

        double distance = computeDistance(features1, features2);  // Euclidean distance
        double similarity = 1.0 / (1.0 + distance);  // Convert to similarity (lower dist = higher weight)

        graph.addEdge(node1, node2, similarity);
    }

    private Pane visualizeGraph(Graph graph) {
        Pane pane = new Pane();
        
        // Draw edges
        for (Edge edge : graph.getEdges()) {
            Line line = new Line(
                edge.getSource().getX(), edge.getSource().getY(),
                edge.getTarget().getX(), edge.getTarget().getY()
            );
            
            double weight = edge.getWeight();
            Color edgeColor = Color.rgb(
                (int)(255 * (1 - weight)),
                (int)(255 * weight),
                128,
                0.7
            );
            
            line.setStroke(edgeColor);
            line.setStrokeWidth(weight * 3 + 0.5);
            pane.getChildren().add(line);
        }
        
        // Draw nodes
        for (Node node : graph.getNodes()) {
            double[] features = node.getFeatures();
            
            // Choose a feature index (e.g., 0 for intensity, or average color/texture)
            double intensity = features.length > 0 ? features[0] : 0.0;
            
            // Normalize intensity to [0, 1]
            intensity = Math.min(1.0, Math.max(0.0, intensity));

            // Use grayscale or color gradient based on intensity
            Color nodeColor = Color.gray(intensity); // Grayscale
            // Or: Color.hsb(intensity * 360, 0.8, 0.9); // For rainbow gradient

            Circle circle = new Circle(
                node.getX(), node.getY(),
                5, // radius
                nodeColor
            );
            circle.setStroke(Color.BLACK);
            pane.getChildren().add(circle);
        }

        
        return pane;
    }

    // Graph Comparison Methods
    public void initiateGraphComparison(Stage parentStage, 
                                      Consumer<Graph> graphLoader,
                                      DoubleConsumer progressUpdater,
                                      Consumer<String> statusUpdater,
                                      Consumer<String> resultConsumer,
                                      Consumer<String> errorHandler) {
        
        Platform.runLater(() -> {
            if (currentGraph == null) {
                errorHandler.accept("Please build a graph first");
                return;
            }

            FileChooser fileChooser = new FileChooser();
            fileChooser.setTitle("Select Image to Compare");
            fileChooser.getExtensionFilters().add(
                new FileChooser.ExtensionFilter("Image Files", "*.png", "*.jpg", "*.jpeg")
            );
            
            File file = fileChooser.showOpenDialog(parentStage);
            if (file != null) {
                executor.execute(() -> {
                    try {
                        Platform.runLater(() -> {
                            progressUpdater.accept(0.0);
                            statusUpdater.accept("Loading comparison image...");
                        });

                        Image compareImage = new Image(file.toURI().toString());
                        Graph secondGraph = createRAGFromImage(compareImage);
                        
                        Platform.runLater(() -> {
                        	progressUpdater.accept(0.5);
                            statusUpdater.accept("Graph created, analyzing...");
                            graphLoader.accept(secondGraph);
                        });

                        double similarity = calculateGraphSimilarity(currentGraph, secondGraph);
                        String resultText = String.format(
                            "=== Graph Comparison Results ===%n" +
                            "Overall Similarity: %.2f%%%n" +
                            "Structural Similarity: %.2f%%%n" +
                            "Color Similarity: %.2f%%",
                            similarity * 100,
                            calculateStructuralSimilarity(currentGraph, secondGraph) * 100,
                            calculateColorSimilarity(currentGraph, secondGraph) * 100
                        );
                        
                        Platform.runLater(() -> {
                            resultConsumer.accept(resultText);
                            statusUpdater.accept("Comparison complete");
                        });
                    } catch (Exception ex) {
                        Platform.runLater(() -> 
                            errorHandler.accept("Error during comparison: " + ex.getMessage())
                        );
                    }
                });
            }
        });
    }

    private double calculateGraphSimilarity(Graph g1, Graph g2) {
        double colorSim = calculateColorSimilarity(g1, g2);
        double structSim = calculateStructuralSimilarity(g1, g2);
        double weightSim = calculateEdgeWeightSimilarity(g1, g2);
        
        return 0.4 * colorSim + 0.3 * structSim + 0.3 * weightSim;
        
    }

    private double calculateEdgeWeightSimilarity(Graph g1, Graph g2) {
        List<Double> weights1 = g1.getEdges().stream()
            .map(Edge::getWeight)
            .sorted()
            .collect(Collectors.toList());
        
        List<Double> weights2 = g2.getEdges().stream()
            .map(Edge::getWeight)
            .sorted()
            .collect(Collectors.toList());

        int size = Math.min(weights1.size(), weights2.size());
        if (size == 0) return 0;
        // T convert to double[] to use existing functionality
        double[] w1 = new double[size];
        double[] w2 = new double[size];
        for (int i = 0; i < size; i++) {
            w1[i] = weights1.get(i);
            w2[i] = weights2.get(i);
        }

        return cosineSimilarity(w1, w2);
    }

    private double calculateColorSimilarity(Graph g1, Graph g2) {
        double[] hist1 = computeColorHistogram(g1);
        double[] hist2 = computeColorHistogram(g2);
        return cosineSimilarity(hist1, hist2);
    }

    private double calculateStructuralSimilarity(Graph g1, Graph g2) {
        double[] vec1 = buildStructuralFeatureVector(g1);
        double[] vec2 = buildStructuralFeatureVector(g2);
        return cosineSimilarity(vec1, vec2);
    }

    private double[] buildStructuralFeatureVector(Graph graph) {
        int totalNodes = graph.getNodes().size();
        if (totalNodes == 0) return new double[12]; // avoid divide-by-zero

        Map<Integer, Integer> degreeDist = calculateDegreeDistribution(graph);

        // Degree histogram (0–9+, 10 bins)
        double[] degreeVec = new double[10];
        for (Map.Entry<Integer, Integer> entry : degreeDist.entrySet()) {
            int degree = Math.min(entry.getKey(), 9); // bucket degree 9+
            degreeVec[degree] += entry.getValue();
        }

        // Normalize degree distribution
        for (int i = 0; i < degreeVec.length; i++) {
            degreeVec[i] /= totalNodes;
        }

        double density = calculateGraphDensity(graph);
        double clustering = calculateAverageClusteringCoefficient(graph);

        // Combine into a single feature vector
        double[] featureVec = Arrays.copyOf(degreeVec, degreeVec.length + 2);
        featureVec[degreeVec.length] = density;
        featureVec[degreeVec.length + 1] = clustering;

        return featureVec;
    }

    private double calculateAverageClusteringCoefficient(Graph graph) {
        Map<Node, Set<Node>> adjacency = new HashMap<>();
        for (Node node : graph.getNodes()) {
            adjacency.put(node, new HashSet<>());
        }
        for (Edge edge : graph.getEdges()) {
            adjacency.get(edge.getSource()).add(edge.getTarget());
            adjacency.get(edge.getTarget()).add(edge.getSource());
        }

        double total = 0.0;
        for (Node node : graph.getNodes()) {
            Set<Node> neighbors = adjacency.get(node);
            int k = neighbors.size();
            if (k < 2) continue;

            int links = 0;
            for (Node ni : neighbors) {
                for (Node nj : neighbors) {
                    if (!ni.equals(nj) && adjacency.get(ni).contains(nj)) {
                        links++;
                    }
                }
            }
            // Each triangle is counted twice
            total += links / (double)(k * (k - 1));
        }

        return total / graph.getNodes().size();
    }

    private Map<Integer, Integer> calculateDegreeDistribution(Graph graph) {
        Map<Integer, Integer> degreeCount = new HashMap<>();
        for (Node node : graph.getNodes()) {
            int degree = graph.getEdges().stream()
                .filter(e -> e.getSource().equals(node) || e.getTarget().equals(node))
                .mapToInt(e -> 1)
                .sum();
            degreeCount.put(degree, degreeCount.getOrDefault(degree, 0) + 1);
        }
        return degreeCount;
    }

    private double[] computeColorHistogram(Graph graph) {
        double[] histogram = new double[64]; // 4x4x4 RGB histogram
        for (Node node : graph.getNodes()) {
            Color color = node.getColor();
            int r = (int) (color.getRed() * 3.999);
            int g = (int) (color.getGreen() * 3.999);
            int b = (int) (color.getBlue() * 3.999);
            int bin = r * 16 + g * 4 + b;
            histogram[bin]++;
        }
        // Normalize
        double sum = Arrays.stream(histogram).sum();
        if (sum > 0) {
            for (int i = 0; i < histogram.length; i++) {
                histogram[i] /= sum;
            }
        }
        return histogram;
    }

    private double cosineSimilarity(double[] a, double[] b) {
        double dot = 0, magA = 0, magB = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            magA += a[i] * a[i];
            magB += b[i] * b[i];
        }
        return magA == 0 || magB == 0 ? 0 : dot / (Math.sqrt(magA) * Math.sqrt(magB));
    }

    // Path Finding Methods
    public void findPathBetweenPoints(double startX, double startY, 
            double endX, double endY,
            Consumer<Graph> pathConsumer,  // Changed from Pane to Graph
            Consumer<String> errorHandler) {
    	
			if (currentGraph == null) {
			errorHandler.accept("No graph available. Please build a graph first.");
			return;
			}
			
			executor.execute(() -> {
			try {
			Node startNode = findNearestNode(currentGraph, startX, startY);
			Node endNode = findNearestNode(currentGraph, endX, endY);
			
			if (startNode == null || endNode == null) {
			throw new Exception("Could not find suitable start or end nodes");
			}
			
			List<Node> path = currentGraph.findShortestPath(startNode, endNode);
			
			if (path.isEmpty()) {
			throw new Exception("No path exists between selected points");
			}
			
			// Create a copy of the graph with path information
			Graph pathGraph = new Graph(currentGraph);
			pathGraph.setPath(path);
			
			Platform.runLater(() -> pathConsumer.accept(pathGraph));
			} catch (Exception e) {
			Platform.runLater(() -> errorHandler.accept("Path finding error: " + e.getMessage()));
			}
			});
	}

    private Node findNearestNode(Graph graph, double x, double y) {
        return graph.getNodes().stream()
            .min(Comparator.comparingDouble(node -> 
                Math.pow(node.getX() - x, 2) + Math.pow(node.getY() - y, 2)))
            .orElse(null);
    }

    // k-NN Classification Methods
    public void trainClassifier(Graph graph, String label) {
        double[] features = extractFeatures(graph);
        trainingData.computeIfAbsent(label, k -> new ArrayList<>())
                   .add(new GraphFeatureVector(features, label));
    }

    public String classifyGraph(Graph targetGraph, int k) {
        if (trainingData.isEmpty()) {
            return "No training data available";
        }

        double[] targetFeatures = extractFeatures(targetGraph);
        PriorityQueue<Neighbor> neighbors = new PriorityQueue<>(Comparator.comparingDouble(n -> n.distance));

        // Find distances to all training examples
        for (Map.Entry<String, List<GraphFeatureVector>> entry : trainingData.entrySet()) {
            for (GraphFeatureVector trainingVector : entry.getValue()) {
                double distance = euclideanDistance(targetFeatures, trainingVector.features);
                neighbors.add(new Neighbor(distance, trainingVector.label));
            }
        }

        // Count votes from k nearest neighbors
        Map<String, Integer> votes = new HashMap<>();
        for (int i = 0; i < k && !neighbors.isEmpty(); i++) {
            String label = neighbors.poll().label;
            votes.put(label, votes.getOrDefault(label, 0) + 1);
        }

        // Return majority vote
        return votes.entrySet().stream()
                   .max(Map.Entry.comparingByValue())
                   .map(Map.Entry::getKey)
                   .orElse("Unknown");
    }

    private static class Neighbor {
        final double distance;
        final String label;

        Neighbor(double distance, String label) {
            this.distance = distance;
            this.label = label;
        }
    }

    private double[] extractFeatures(Graph graph) {
    	 int nodeCount = graph.getNodes().size();
    	    int edgeCount = graph.getEdges().size();
    	    double avgDegree = calculateAverageDegree(nodeCount, edgeCount);
    	    double colorVariance = calculateColorVariance(graph);
    	    double density = calculateGraphDensity(nodeCount, edgeCount);
    	    double clustering = calculateAverageClustering(graph);
    	    double avgEdgeWeight = calculateAverageEdgeWeight(graph);

    	    return new double[] {
    	        nodeCount,
    	        edgeCount,
    	        avgDegree,
    	        colorVariance,
    	        density,
    	        clustering,
    	        avgEdgeWeight
    	    };
    }
    private double calculateAverageEdgeWeight(Graph graph) {
        if (graph.getEdges().isEmpty()) return 0;
        return graph.getEdges().stream()
            .mapToDouble(Edge::getWeight)
            .average()
            .orElse(0);
    }

    private double calculateAverageDegree(int nodeCount, int edgeCount) {
        return nodeCount == 0 ? 0 : (2.0 * edgeCount) / nodeCount;
    }


    private double calculateColorVariance(Graph graph) {
        if (graph.getNodes().isEmpty()) return 0;

        double avgR = graph.getNodes().stream().mapToDouble(n -> n.getColor().getRed()).average().orElse(0);
        double avgG = graph.getNodes().stream().mapToDouble(n -> n.getColor().getGreen()).average().orElse(0);
        double avgB = graph.getNodes().stream().mapToDouble(n -> n.getColor().getBlue()).average().orElse(0);

        return graph.getNodes().stream().mapToDouble(n -> {
            double dr = n.getColor().getRed() - avgR;
            double dg = n.getColor().getGreen() - avgG;
            double db = n.getColor().getBlue() - avgB;
            return dr * dr + dg * dg + db * db;
        }).average().orElse(0);
    }

    private double calculateGraphDensity(int nodeCount, int edgeCount) {
        if (nodeCount < 2) return 0;
        int maxEdges = nodeCount * (nodeCount - 1) / 2;
        return maxEdges == 0 ? 0 : (double) edgeCount / maxEdges;
    }
    private double calculateGraphDensity(Graph graph) {
        return calculateGraphDensity(graph.getNodes().size(), graph.getEdges().size());
    }


    private double calculateAverageClustering(Graph graph) {
        double totalClustering = 0;
        int counted = 0;

        for (Node node : graph.getNodes()) {
            Set<Node> neighbors = getNeighbors(graph, node);
            int k = neighbors.size();
            if (k < 2) continue;

            int links = 0;
            List<Node> neighborList = new ArrayList<>(neighbors);
            for (int i = 0; i < neighborList.size(); i++) {
                for (int j = i + 1; j < neighborList.size(); j++) {
                    Node n1 = neighborList.get(i);
                    Node n2 = neighborList.get(j);
                    if (areConnected(graph, n1, n2)) links++;
                }
            }

            double possibleLinks = k * (k - 1) / 2.0;
            totalClustering += links / possibleLinks;
            counted++;
        }

        return counted == 0 ? 0 : totalClustering / counted;
    }

    private Set<Node> getNeighbors(Graph graph, Node node) {
        Set<Node> neighbors = new HashSet<>();
        for (Edge edge : graph.getEdges()) {
            if (edge.getSource().equals(node)) neighbors.add(edge.getTarget());
            else if (edge.getTarget().equals(node)) neighbors.add(edge.getSource());
        }
        return neighbors;
    }

    private boolean areConnected(Graph graph, Node a, Node b) {
        return graph.getEdges().stream().anyMatch(e ->
            (e.getSource().equals(a) && e.getTarget().equals(b)) ||
            (e.getSource().equals(b) && e.getTarget().equals(a))
        );
    }



 /*   private Set<Node> getNeighbors(Graph graph, Node node) {
        return graph.getEdges().stream()
            .filter(e -> e.getSource().equals(node) || e.getTarget().equals(node))
            .map(e -> e.getSource().equals(node) ? e.getTarget() : e.getSource())
            .collect(Collectors.toSet());
    }
*/
    private double euclideanDistance(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(sum);
    }

    // Utility Methods
    private double colorDistance(Color c1, Color c2) {
        return Math.sqrt(
            Math.pow(c1.getRed() - c2.getRed(), 2) +
            Math.pow(c1.getGreen() - c2.getGreen(), 2) +
            Math.pow(c1.getBlue() - c2.getBlue(), 2)
        ) / Math.sqrt(3);
    }
    
    //graph method
    //RAG
  /*  public void buildRegionAdjacencyGraph(List<Region> regions, int[][] adjacencyMatrix) {
        currentGraph = new Graph();
        Map<Integer, Node> regionNodes = new HashMap<>();

        for (Region r : regions) {
            RegionFeatures rf = new RegionFeatures(r);
            Node node = new Node(r.centroidX, r.centroidY);
            node.setFeatures(rf.extractFeatures());
            currentGraph.addNode(node);
            regionNodes.put(r.id, node);
        }

        for (int i = 0; i < regions.size(); i++) {
            for (int j = 0; j < regions.size(); j++) {
                if (i != j && adjacencyMatrix[i][j] == 1) {
                    Region r1 = regions.get(i);
                    Region r2 = regions.get(j);
                    r1.addNeighbor(r2.id);
                    r2.addNeighbor(r1.id);

                    Node n1 = regionNodes.get(r1.id);
                    Node n2 = regionNodes.get(r2.id);
                    double similarity = RegionFeatures.computeSimilarity(r1, r2);
                    currentGraph.addEdge(n1, n2, similarity);
                }
            }
        }
    }*/


    // --- K-NN Graph ---
    public void buildKNNGraph(int k) {
        if (currentGraph == null) return;
        List<Node> nodes = currentGraph.getNodes();

        for (Node n : nodes) {
            PriorityQueue<NodeDistance> pq = new PriorityQueue<>(Comparator.comparingDouble(nd -> nd.distance));
            for (Node other : nodes) {
                if (n == other) continue;
                double dist = computeDistance(n.getFeatures(), other.getFeatures());
                pq.add(new NodeDistance(other, dist));
            }

            for (int i = 0; i < k && !pq.isEmpty(); i++) {
                NodeDistance nd = pq.poll();
                double similarity = 1.0 / (1.0 + nd.distance);
                currentGraph.addEdge(n, nd.node, similarity);
            }
        }
    }


    private double computeDistance(double[] a, double[] b) {
        double sum=0;
        for (int i=0; i<a.length; i++) {
            sum += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(sum);
    }

    private static class NodeDistance {
        Node node;
        double distance;
        NodeDistance(Node node, double dist) {
            this.node = node; this.distance=dist;
        }
    }

    // --- MST (Prim's algorithm) ---
    public void computeMST() {
        if (currentGraph == null || currentGraph.getNodes().isEmpty()) return;

        Graph mst = new Graph();
        Set<Node> visited = new HashSet<>();
        PriorityQueue<Edge> pq = new PriorityQueue<>(Comparator.comparingDouble(Edge::getWeight));

        Node start = currentGraph.getNodes().get(0);
        visited.add(start);
        mst.addNode(start);
        pq.addAll(currentGraph.getEdgesFrom(start));

        while (!pq.isEmpty()) {
            Edge e = pq.poll();
            Node u = e.getSource();
            Node v = e.getTarget();

            Node nextNode = !visited.contains(v) ? v : (!visited.contains(u) ? u : null);
            if (nextNode == null) continue;

            visited.add(nextNode);
            mst.addNode(nextNode);
            mst.addEdge(u, v, e.getWeight());

            for (Edge neighborEdge : currentGraph.getEdgesFrom(nextNode)) {
                if (!visited.contains(neighborEdge.getTarget()) || !visited.contains(neighborEdge.getSource())) {
                    pq.add(neighborEdge);
                }
            }
        }

        currentGraph = mst;
    }


    public void shutdown() {
        executor.shutdownNow();
    }

    public Image getCurrentImage() {
        return currentImage;
    }

    public Graph getCurrentGraph() {
        return currentGraph;
    }
}