package Gui;

import Implementations.AppLogic;
import graph.Edge;
import graph.Graph;
import graph.Node;
import javafx.application.Platform;
import javafx.event.EventHandler;
import javafx.geometry.BoundingBox;
import javafx.geometry.Bounds;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.effect.DropShadow;
import javafx.scene.image.*;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
import javafx.scene.shape.Line;
import javafx.scene.shape.StrokeLineCap;
import javafx.stage.Screen;
import javafx.stage.Stage;
import javafx.util.Pair;

public class UserInterface {
    private final Stage primaryStage;
    private final AppLogic appLogic;
    
    private ImageView imageView;
    private TextArea resultsArea;
    private ProgressBar progressBar;
    private Label statusLabel;
    private ScrollPane graphScrollPane;
    private Pane graphPane;
    private TabPane tabPane;
    private GridPane comparisonPane;
    private Button classifyButton;
    private double initialWidth = 1200;
    private double initialHeight = 700;
    private Pane originalGraphPane;
    private Pane comparisonGraphPane;
	private Pane overlayPane;
	private SplitPane mainSplitPane;
	
    // Color scheme
    private static final Color PRIMARY_COLOR = Color.rgb(0, 120, 215);
    private static final Color SECONDARY_COLOR = Color.rgb(100, 180, 255);
    private static final Color PATH_COLOR = Color.rgb(255, 0, 0);
    private static final Color START_NODE_COLOR = Color.rgb(0, 200, 0);
    private static final Color END_NODE_COLOR = Color.rgb(0, 0, 200);
    
    //Fields original coordinates of points
    private double startPointOriginalX = -1;
    private double startPointOriginalY = -1;
    private double endPointOriginalX = -1;
    private double endPointOriginalY = -1;

    public UserInterface(Stage primaryStage, AppLogic appLogic) {
        this.primaryStage = primaryStage;
        this.appLogic = appLogic;
        appLogic.setParentWindow(primaryStage);
        
        this.originalGraphPane = new Pane();
        this.comparisonGraphPane = new Pane();
        this.overlayPane=new Pane();
        
        initialize();
    }

    public void initialize() {
        Screen screen = Screen.getPrimary();
        double screenWidth = screen.getVisualBounds().getWidth();
        double screenHeight = screen.getVisualBounds().getHeight();
        
        initialWidth = Math.min(screenWidth * 0.8, 1600);
        initialHeight = Math.min(screenHeight * 0.8, 900);
        
        BorderPane root = new BorderPane();
        root.setTop(createMenuBar());
        root.setCenter(createContentPane());
        root.setBottom(createStatusBar());
       /* Image icon=new Image("icon,jpg");
        primaryStage.getIcons().add(icon);*/
        Scene scene = new Scene(root, initialWidth, initialHeight);
        primaryStage.setScene(scene);
        primaryStage.setTitle("Graph-Based Image Analyzer");
        
        primaryStage.setMinWidth(800);
        primaryStage.setMinHeight(600);
        primaryStage.centerOnScreen();
        primaryStage.show();
        
        graphScrollPane.viewportBoundsProperty().addListener((obs, oldBounds, newBounds) -> {
            fitGraphToView();
        });
        
        primaryStage.maximizedProperty().addListener((obs, oldVal, newVal) -> {
            Platform.runLater(this::enforceDividerPosition);
        });
        
        // Add resize listener
        primaryStage.widthProperty().addListener((obs, oldVal, newVal) -> {
            Platform.runLater(this::enforceDividerPosition);
        });
        
        setOnCloseRequest();
    }
    
    
    private void enforceDividerPosition() {
        Platform.runLater(() -> {
            if (mainSplitPane != null) {
                mainSplitPane.setDividerPositions(0.05);
            }
        });
    }

    private MenuBar createMenuBar() {
        MenuBar menuBar = new MenuBar();
        menuBar.setStyle("-fx-background-color: " + toHex(PRIMARY_COLOR) + ";");
        
        Menu fileMenu = new Menu("File");
        fileMenu.setStyle("-fx-text-fill: white;");
        MenuItem openItem = new MenuItem("Open Image");
        openItem.setOnAction(e -> handleOpenImage());
        MenuItem exitItem = new MenuItem("Exit");
        exitItem.setOnAction(e -> primaryStage.close());
        fileMenu.getItems().addAll(openItem, new SeparatorMenuItem(), exitItem);
        
        Menu analysisMenu = new Menu("Analysis");
        analysisMenu.setStyle("-fx-text-fill: white;");
        MenuItem buildGraphItem = new MenuItem("Build RAG");
        buildGraphItem.setOnAction(e -> handleBuildGraph());
        MenuItem compareItem = new MenuItem("Compare Images");
        compareItem.setOnAction(e -> handleCompareGraphs());
        MenuItem pathItem = new MenuItem("Find Path");
        pathItem.setOnAction(e -> handleFindPath());
        MenuItem classifyItem = new MenuItem("Classify Image");
        classifyItem.setOnAction(e -> handleClassifyImage());
        MenuItem zoomFitItem = new MenuItem("Zoom to Fit");
        zoomFitItem.setOnAction(e -> fitGraphToView());
        analysisMenu.getItems().addAll(buildGraphItem, compareItem, 
                new SeparatorMenuItem(), pathItem, classifyItem,
                new SeparatorMenuItem(), zoomFitItem);

        
        Menu buildmenu=new Menu("Build");
        MenuItem buildKNNItem = new MenuItem("Build KNN Graph");
        buildKNNItem.setOnAction(e -> handleBuildKNN());

        MenuItem computeMSTItem = new MenuItem("Compute MST");
        computeMSTItem.setOnAction(e -> handleComputeMST());
        
        buildmenu.getItems().addAll(buildKNNItem,computeMSTItem);
        
     
        menuBar.getMenus().addAll(fileMenu, analysisMenu,buildmenu);
        return menuBar;
    }

	private SplitPane createContentPane() {
		mainSplitPane = new SplitPane();
        mainSplitPane.setDividerPositions(0.05);
        
        VBox leftPanel = new VBox(15);
        leftPanel.setPadding(new Insets(15));
        leftPanel.setAlignment(Pos.TOP_CENTER);
        leftPanel.setStyle("-fx-background-color: #f5f5f5;");
        
        Label imageLabel = new Label("SOURCE IMAGE");
        imageLabel.setStyle("-fx-font-size: 14; -fx-font-weight: bold; -fx-text-fill: #333;");
        
        imageView = new ImageView();
        imageView.setPreserveRatio(true);
        imageView.setSmooth(true);
        imageView.setFitWidth(300);
        imageView.setStyle("-fx-effect: dropshadow(gaussian, rgba(0,0,0,0.2), 10, 0, 0, 0);");
        
        VBox buttonBox = new VBox(8);
        buttonBox.setAlignment(Pos.CENTER);
        buttonBox.setPadding(new Insets(10, 0, 10, 0));
        
        Button openButton = createStyledButton("Open Image", PRIMARY_COLOR, e -> handleOpenImage());
        Button buildGraphButton = createStyledButton("Build Graph", PRIMARY_COLOR, e -> handleBuildGraph());
        Button compareButton = createStyledButton("Compare Images", PRIMARY_COLOR, e -> handleCompareGraphs());
        Button findPathButton = createStyledButton("Find Path", PRIMARY_COLOR, e -> handleFindPath());
        classifyButton = createStyledButton("Classify Image", PRIMARY_COLOR, e -> handleClassifyImage());
        classifyButton.setDisable(true);
        
        buttonBox.getChildren().addAll(openButton, buildGraphButton, compareButton, 
                                     findPathButton, classifyButton);
        
        progressBar = new ProgressBar(0);
        progressBar.setStyle("-fx-accent: " + toHex(PRIMARY_COLOR) + ";");
        progressBar.setMaxWidth(Double.MAX_VALUE);
        
        leftPanel.getChildren().addAll(imageLabel, imageView, buttonBox, progressBar);
        
        tabPane = new TabPane();
        tabPane.setStyle("-fx-background-color: white;");
        
        graphPane = new Pane();
        graphPane.setStyle("-fx-background-color: white;");
        
        graphScrollPane = new ScrollPane(graphPane);
        graphScrollPane.setFitToWidth(true);
        graphScrollPane.setFitToHeight(true);
        graphScrollPane.setStyle("-fx-background: white;");
        
        //Center the content in the ScrollPane
        graphScrollPane.setContent(graphPane);
        graphPane.translateXProperty().bind(
            graphScrollPane.widthProperty()
                .subtract(graphPane.widthProperty())
                .divide(2)
        );
        
        graphPane.translateYProperty().bind(
            graphScrollPane.heightProperty()
                .subtract(graphPane.heightProperty())
                .divide(2)
        );
        
        Tab graphTab = new Tab("GRAPH VISUALIZATION", graphScrollPane);
        graphTab.setClosable(false);
        graphTab.setStyle("-fx-font-weight: bold;");
        
        resultsArea = new TextArea();
        resultsArea.setEditable(false);
        resultsArea.setWrapText(true);
        resultsArea.setStyle("-fx-font-family: 'Segoe UI'; -fx-font-size: 13;");
        ScrollPane resultsScroll = new ScrollPane(resultsArea);
        resultsScroll.setFitToWidth(true);
        resultsScroll.setFitToHeight(true);
        
        Tab resultsTab = new Tab("ANALYSIS RESULTS", resultsScroll);
        resultsTab.setClosable(false);
        resultsTab.setStyle("-fx-font-weight: bold;");
        
        comparisonPane = createComparisonPane();
        Tab comparisonTab = new Tab("GRAPH COMPARISON", comparisonPane);
        comparisonTab.setClosable(false);
        comparisonTab.setStyle("-fx-font-weight: bold;");
        
        tabPane.getTabs().addAll(graphTab, resultsTab, comparisonTab);
        
        mainSplitPane.getItems().addAll(leftPanel, tabPane);
        return mainSplitPane;
    }

    private GridPane createComparisonPane() {
        GridPane pane = new GridPane();
        pane.setPadding(new Insets(15));
        pane.setHgap(20);
        pane.setVgap(20);
        pane.setAlignment(Pos.CENTER);
        pane.setStyle("-fx-background-color: white;");
        
        VBox originalBox = new VBox(10);
        originalBox.setAlignment(Pos.CENTER);
        
        Label originalLabel = new Label("ORIGINAL GRAPH");
        originalLabel.setStyle("-fx-font-size: 14; -fx-font-weight: bold; -fx-text-fill: " + toHex(PRIMARY_COLOR) + ";");
        
        originalGraphPane = new StackPane();
        originalGraphPane.setStyle("-fx-background-color: white");
        
        ScrollPane originalScroll = new ScrollPane(originalGraphPane);
        originalScroll.setFitToWidth(true);
        originalScroll.setFitToHeight(true);
        originalScroll.setPrefViewportWidth(600);
        originalScroll.setPrefViewportHeight(300);
        originalScroll.setStyle("-fx-background: white;");
        
        originalBox.getChildren().addAll(originalLabel, originalScroll);
        
        VBox compareBox = new VBox(10);
        compareBox.setAlignment(Pos.CENTER);
        
        Label compareLabel = new Label("COMPARISON GRAPH");
        compareLabel.setStyle("-fx-font-size: 14; -fx-font-weight: bold; -fx-text-fill: " + toHex(SECONDARY_COLOR) + ";");
        
        comparisonGraphPane = new StackPane();
        comparisonGraphPane.setStyle("-fx-background-color: white");
        
        
        ScrollPane compareScroll = new ScrollPane(comparisonGraphPane);
        compareScroll.setFitToWidth(true);
        compareScroll.setFitToHeight(true);
        compareScroll.setPrefViewportWidth(600);
        compareScroll.setPrefViewportHeight(300);
        compareScroll.setStyle("-fx-background: white;");
        
        compareBox.getChildren().addAll(compareLabel, compareScroll);
        
        pane.add(originalBox, 0, 0);
        pane.add(compareBox, 0, 1);
        
        return pane;
    }

    private HBox createStatusBar() {
        statusLabel = new Label("Ready");
        statusLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 14;");
        
        HBox statusBar = new HBox(10, new Label("Status:"), statusLabel);
        statusBar.setAlignment(Pos.CENTER_LEFT);
        statusBar.setPadding(new Insets(10, 15, 10, 15));
        statusBar.setStyle("-fx-background-color: #f5f5f5; -fx-border-color: #e0e0e0; -fx-border-width: 1 0 0 0;");
        return statusBar;
    }
    
    private void handleComputeMST() {
    	appLogic.computeMST();
	    // Visualize
	    graphPane.getChildren().clear();
	    Pane visualization = createGraphVisualization(appLogic.getCurrentGraph());
	    graphPane.getChildren().add(visualization);
	    fitGraphToView();
	    appendResults("MST computed and visualized\n");
	}

	private void handleBuildKNN() {
		int k=5;
		 appLogic.buildKNNGraph(k);
		    // Visualize
		    graphPane.getChildren().clear();
		    Pane visualization = createGraphVisualization(appLogic.getCurrentGraph());
		    graphPane.getChildren().add(visualization);
		    fitGraphToView();
		    appendResults("KNN graph built with k=" + k + "\n");
	
	}
    
    private Pane createGraphVisualization(Graph graph) {
        Pane visualizationPane = new Pane();
        visualizationPane.setStyle(
                "-fx-background-color: white;" +
                "-fx-border-radius: 5;" +
                "-fx-background-radius: 5;" +
                "-fx-effect: dropshadow(gaussian, rgba(0,0,0,0.1), 5, 0, 0, 1);"
            );
        
        // Calculate bounds with padding
        double padding = 20;
        Bounds bounds = calculateGraphBounds(graph, padding);
        
        // Set pane size to match graph dimensions
        visualizationPane.setMinSize(bounds.getWidth(), bounds.getHeight());
        visualizationPane.setPrefSize(bounds.getWidth(), bounds.getHeight());
        
        // Draw elements with offset to account for padding
        for (Edge edge : graph.getEdges()) {
            Line line = new Line(
                edge.getSource().getX() - bounds.getMinX(),
                edge.getSource().getY() - bounds.getMinY(),
                edge.getTarget().getX() - bounds.getMinX(),
                edge.getTarget().getY() - bounds.getMinY()
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
            line.setStrokeLineCap(StrokeLineCap.ROUND);
            visualizationPane.getChildren().add(line);
        }
        
        for (Node node : graph.getNodes()) {
            Circle circle = new Circle(
                node.getX() - bounds.getMinX(),
                node.getY() - bounds.getMinY(),
                5,
                node.getColor()
            );
            circle.setStroke(Color.BLACK);
            circle.setStrokeWidth(1);
            circle.setEffect(new DropShadow(3, Color.gray(0.3)));
            visualizationPane.getChildren().add(circle);
        }
        
        // Set up view and interactions
        fitGraphToView();
        addZoomAndPanSupport(visualizationPane);
        
        return visualizationPane;
    }
    
    private Bounds calculateGraphBounds(Graph graph, double padding) {
        double minX = Double.MAX_VALUE, minY = Double.MAX_VALUE;
        double maxX = Double.MIN_VALUE, maxY = Double.MIN_VALUE;
        
        for (Node node : graph.getNodes()) {
            minX = Math.min(minX, node.getX());
            minY = Math.min(minY, node.getY());
            maxX = Math.max(maxX, node.getX());
            maxY = Math.max(maxY, node.getY());
        }
        
        // Apply padding
        minX -= padding;
        minY -= padding;
        maxX += padding;
        maxY += padding;
        
        // Ensure minimum size
        double width = Math.max(maxX - minX, 100);
        double height = Math.max(maxY - minY, 100);
        
        return new BoundingBox(minX, minY, width, height);
    }
    
    private Pane createPathVisualization(Graph graph) {
        Pane visualization = createGraphVisualization(graph);
        
        if (graph.getPath() != null && !graph.getPath().isEmpty()) {
            for (int i = 0; i < graph.getPath().size() - 1; i++) {
                Node current = graph.getPath().get(i);
                Node next = graph.getPath().get(i+1);
                
                Line pathLine = new Line(
                    current.getX(), current.getY(),
                    next.getX(), next.getY()
                );
                pathLine.setStroke(PATH_COLOR);
                pathLine.setStrokeWidth(3);
                pathLine.setStrokeLineCap(StrokeLineCap.ROUND);
                visualization.getChildren().add(pathLine);
            }
            
            Circle startNode = new Circle(
                graph.getPath().get(0).getX(),
                graph.getPath().get(0).getY(),
                10,
                START_NODE_COLOR
            );
            startNode.setStroke(Color.WHITE);
            startNode.setStrokeWidth(2);
            
            Circle endNode = new Circle(
                graph.getPath().get(graph.getPath().size()-1).getX(),
                graph.getPath().get(graph.getPath().size()-1).getY(),
                10,
                END_NODE_COLOR
            );
            endNode.setStroke(Color.WHITE);
            endNode.setStrokeWidth(2);
            
            visualization.getChildren().addAll(startNode, endNode);
        }
        
        // Set up view and interactions
        fitGraphToView();
        addZoomAndPanSupport(visualization);
        
        return visualization;
    }
    
    private void handleOpenImage() {
        appLogic.handleOpenImage(
            image -> Platform.runLater(() -> {
                updateImageView(image);
                graphPane.getChildren().clear();
                comparisonGraphPane.getChildren().clear();
                classifyButton.setDisable(false);
            }),
            this::showError
        );
    }
    
    private void handleBuildGraph() {
        if (imageView.getImage() == null) {
            showError("Please load an image first");
            return;
        }
        
        progressBar.setProgress(0);
        statusLabel.setText("Building graph...");
        
        appLogic.handleBuildGraph(
            progress -> Platform.runLater(() -> progressBar.setProgress(progress)),
            graph -> Platform.runLater(() -> {
                try {
                    graphPane.getChildren().clear();
                    
                    // Create visualization and add to graphPane
                    Pane visualization = createGraphVisualization(graph);
                    graphPane.getChildren().add(visualization);
                    
                    // Update UI
                    tabPane.getSelectionModel().select(0);
                    updateStatus("Graph built successfully");
                } catch (Exception e) {
                    showError("Error displaying graph: " + e.getMessage());
                    e.printStackTrace();
                }
            }),
            error -> Platform.runLater(() -> showError(error))
        );
    }

    
    private void fitGraphToView() {
    	if (graphPane.getChildren().isEmpty()) return;
        
        Pane content = (Pane) graphPane.getChildren().get(0);
        Bounds bounds = content.getBoundsInParent();
        
        double graphWidth = bounds.getWidth();
        double graphHeight = bounds.getHeight();
        double viewWidth = graphScrollPane.getViewportBounds().getWidth();
        double viewHeight = graphScrollPane.getViewportBounds().getHeight();
        
        if (graphWidth <= 0 || graphHeight <= 0) return;
        
        double scale = Math.min(
            viewWidth / graphWidth,
            viewHeight / graphHeight
        ) * 0.9; // 10% margin
        
        content.setScaleX(scale);
        content.setScaleY(scale);
        content.setTranslateX(0);
        content.setTranslateY(0);
        
        Platform.runLater(() -> {
            graphScrollPane.setHvalue(0.5);
            graphScrollPane.setVvalue(0.5);
        });
    }
    
    private void addZoomAndPanSupport(Pane pane) {
        pane.setOnScroll(event -> {
            double zoomFactor = event.getDeltaY() > 0 ? 1.1 : 0.9;
            pane.setScaleX(pane.getScaleX() * zoomFactor);
            pane.setScaleY(pane.getScaleY() * zoomFactor);
            event.consume();
        });
        
        final double[] dragStartX = new double[1];
        final double[] dragStartY = new double[1];
        
        pane.setOnMousePressed(event -> {
            dragStartX[0] = event.getSceneX() - pane.getTranslateX();
            dragStartY[0] = event.getSceneY() - pane.getTranslateY();
        });
        
        pane.setOnMouseDragged(event -> {
            pane.setTranslateX(event.getSceneX() - dragStartX[0]);
            pane.setTranslateY(event.getSceneY() - dragStartY[0]);
        });
    }
    
    private void handleCompareGraphs() {
        if (imageView.getImage() == null) {
            showError("Please load an image first");
            return;
        }
        
        // Clear previous results and prepare UI
        progressBar.setProgress(0);
        statusLabel.setText("Preparing comparison...");
        originalGraphPane.getChildren().clear();
        comparisonGraphPane.getChildren().clear();
        
        // Create and display visualization for original graph
        Pane originalVisualization = createGraphVisualization(appLogic.getCurrentGraph());
        originalGraphPane.getChildren().add(originalVisualization);
        
        // Initiate the comparison process
        appLogic.initiateGraphComparison(
            primaryStage,
            comparisonGraph -> Platform.runLater(() -> {
                try {
                	// Visualize the second graph
                    Pane comparisonVisualization = createGraphVisualization(comparisonGraph);
                    comparisonGraphPane.getChildren().add(comparisonVisualization);
                    
                    // Fit both graphs to view after layout pass
                    Platform.runLater(() -> {
                        ScrollPane originalScroll = (ScrollPane)((VBox)comparisonPane.getChildren().get(0)).getChildren().get(1);
                        ScrollPane compareScroll = (ScrollPane)((VBox)comparisonPane.getChildren().get(1)).getChildren().get(1);
                        
                        fitGraphToPane(originalGraphPane, originalScroll);
                        fitGraphToPane(comparisonGraphPane, compareScroll);
                    });
                    
                    // Switch to comparison tab
                    tabPane.getSelectionModel().select(2);
                } catch (Exception e) {
                    showError("Error displaying comparison: " + e.getMessage());
                }
            }),
            progress -> progressBar.setProgress(progress),
            status -> statusLabel.setText(status),
            result -> {
                appendResults("\n=== COMPARISON RESULTS ===\n");
                appendResults(result + "\n");
                updateStatus("Comparison complete");
            },
            error -> showError(error)
        );
    }

    
    private void fitGraphToPane(Pane graphContent, ScrollPane container) {
        if (graphContent.getChildren().isEmpty()) return;
        
        Pane content = (Pane)graphContent.getChildren().get(0);
        Bounds bounds = content.getBoundsInLocal();
        
        double graphWidth = bounds.getWidth();
        double graphHeight = bounds.getHeight();
        double viewWidth = container.getViewportBounds().getWidth();
        double viewHeight = container.getViewportBounds().getHeight();
        
        if (graphWidth <= 0 || graphHeight <= 0) return;
        
        // Calculate scale to fit with 10% margin
        double scale = Math.min(
            viewWidth / graphWidth,
            viewHeight / graphHeight
        ) * 0.9;
        
        content.setScaleX(scale);
        content.setScaleY(scale);
        
        // Center the content in the viewport
        Platform.runLater(() -> {
            container.setHvalue(0.5);
            container.setVvalue(0.5);
            
            // Force layout update
            container.layout();
            container.requestLayout();
        });
    }
    
    private void handleFindPath() {
        if (graphPane.getChildren().isEmpty()) {
            showError("Please build a graph first");
            return;
        }
        
        Dialog<Pair<Double, Double>> dialog = new Dialog<>();
        dialog.setTitle("Path Finding");
        dialog.setHeaderText("Select start and end points on the image.\n"
                + "Click once to set the START point (green).\n"
                + "Click again to set the END point (red).\n"
                + "You can click multiple times to reset points.\n"
                + "After selecting both points, click 'Find Path'.");
        
        ButtonType findButtonType = new ButtonType("Find Path", ButtonBar.ButtonData.OK_DONE);
        dialog.getDialogPane().getButtonTypes().addAll(findButtonType, ButtonType.CANCEL);
        
        ImageView pathImageView = new ImageView(imageView.getImage());
        pathImageView.setPreserveRatio(true);
        pathImageView.setFitWidth(400);
        
        Circle startPoint = new Circle(5, Color.TRANSPARENT);
        startPoint.setStroke(Color.GREEN);
        startPoint.setStrokeWidth(2);
        Circle endPoint = new Circle(5, Color.TRANSPARENT);
        endPoint.setStroke(Color.RED);
        endPoint.setStrokeWidth(2);
        
        Pane selectionPane = new Pane(pathImageView, startPoint, endPoint);
        selectionPane.setPrefSize(400, 400);
        
        final int[] clickCount = {0};
        
        selectionPane.setOnMouseClicked(e -> {
            double displayedWidth = pathImageView.getBoundsInParent().getWidth();
            double displayedHeight = pathImageView.getBoundsInParent().getHeight();

            double originalWidth = pathImageView.getImage().getWidth();
            double originalHeight = pathImageView.getImage().getHeight();

            double scaleX = originalWidth / displayedWidth;
            double scaleY = originalHeight / displayedHeight;

            double originalX = e.getX() * scaleX;
            double originalY = e.getY() * scaleY;

            if (clickCount[0] % 2 == 0) {
                // Set start point (green)
                startPoint.setCenterX(e.getX());
                startPoint.setCenterY(e.getY());
                startPoint.setFill(Color.GREEN.deriveColor(0, 1, 1, 0.3));

                startPointOriginalX = originalX;
                startPointOriginalY = originalY;
            } else {
                // Set end point (red)
                endPoint.setCenterX(e.getX());
                endPoint.setCenterY(e.getY());
                endPoint.setFill(Color.RED.deriveColor(0, 1, 1, 0.3));

                endPointOriginalX = originalX;
                endPointOriginalY = originalY;
            }
            clickCount[0]++;
        });
        
        dialog.getDialogPane().setContent(selectionPane);
        
        dialog.setResultConverter(dialogButton -> {
            if (dialogButton == findButtonType) {
                // Check if both points are selected
                if (startPointOriginalX == -1 || startPointOriginalY == -1 || endPointOriginalX == -1 || endPointOriginalY == -1) {
                    showError("Please select both start and end points before finding path.");
                    return null;
                }
                return new Pair<>(startPointOriginalX, startPointOriginalY);
            }
            return null;
        });
        
        dialog.showAndWait().ifPresent(startCoords -> {
            double startX = startCoords.getKey();
            double startY = startCoords.getValue();

            double endX = endPointOriginalX;
            double endY = endPointOriginalY;

            progressBar.setProgress(0);
            statusLabel.setText("Finding path...");
            
            appLogic.findPathBetweenPoints(
                startX, startY,
                endX, endY,
                pathGraph -> Platform.runLater(() -> {
                    graphPane.getChildren().clear();
                    Pane pathVisualization = createPathVisualization(pathGraph);
                    graphPane.getChildren().add(pathVisualization);
                    fitGraphToView();
                    tabPane.getSelectionModel().select(0);
                    appendResults("=== Path Found ===\n");
                    appendResults("Start: (" + (int)startX + ", " + (int)startY + ")\n");
                    appendResults("End: (" + (int)endX + ", " + (int)endY + ")\n");
                    updateStatus("Path found successfully");
                }),
                error -> Platform.runLater(() -> showError(error))
            );
        });
    }
    
    private void handleClassifyImage() {
        if (imageView.getImage() == null) {
            showError("Please load an image first");
            return;
        }
        
        TextInputDialog dialog = new TextInputDialog();
        dialog.setTitle("Image Classification");
        dialog.setHeaderText("Enter training label for this image");
        dialog.setContentText("Label:");
        
        dialog.showAndWait().ifPresent(label -> {
            appLogic.trainClassifier(appLogic.getCurrentGraph(), label);
            appendResults("Image trained as: " + label + "\n");
            
            String classification = appLogic.classifyGraph(appLogic.getCurrentGraph(), 3);
            appendResults("Classification result: " + classification + "\n");
            updateStatus("Classification complete");
        });
    }
    
    private void updateImageView(Image image) {
        imageView.setImage(image);
        appendResults("=== New Image Loaded ===\n");
        appendResults("Dimensions: " + (int)image.getWidth() + "x" + (int)image.getHeight() + "\n");
        updateStatus("Image loaded successfully");
    }
    
    private void updateStatus(String message) {
        statusLabel.setText(message);
        appendResults("[Status] " + message + "\n");
    }
    
    private void appendResults(String text) {
        resultsArea.appendText(text);
        resultsArea.setScrollTop(Double.MAX_VALUE);
    }
    
    private void showError(String error) {
        Alert alert = new Alert(Alert.AlertType.ERROR);
        alert.setTitle("Error");
        alert.setHeaderText("Operation Failed");
        alert.setContentText(error);
        alert.showAndWait();
        appendResults("!!! ERROR: " + error + "\n");
        updateStatus("Error occurred");
    }

    private Button createStyledButton(String text, Color color, EventHandler<javafx.event.ActionEvent> handler) {
        Button button = new Button(text);
        button.setOnAction(handler);
        button.setStyle("-fx-background-color: " + toHex(color) + 
                      "; -fx-text-fill: white; -fx-font-weight: bold;" +
                      "; -fx-font-size: 13; -fx-padding: 8 15;");
        button.setMaxWidth(Double.MAX_VALUE);
        
        button.setOnMouseEntered(e -> button.setStyle("-fx-background-color: " + toHex(color.brighter()) + 
                                                   "; -fx-text-fill: white; -fx-font-weight: bold;" +
                                                   "; -fx-font-size: 13; -fx-padding: 8 15;" +
                                                   "-fx-effect: dropshadow(gaussian, " + toHex(color.brighter()) + ", 8, 0, 0, 0);"));
        button.setOnMouseExited(e -> button.setStyle("-fx-background-color: " + toHex(color) + 
                                                  "; -fx-text-fill: white; -fx-font-weight: bold;" +
                                                  "; -fx-font-size: 13; -fx-padding: 8 15;" +
                                                  "-fx-effect: null;"));
        
        return button;
    }

    private String toHex(Color color) {
        return String.format("#%02X%02X%02X",
            (int)(color.getRed() * 255),
            (int)(color.getGreen() * 255),
            (int)(color.getBlue() * 255));
    }

    private void setOnCloseRequest() {
        primaryStage.setOnCloseRequest(e -> {
            appLogic.shutdown();
            Platform.exit();
        });
    }
}