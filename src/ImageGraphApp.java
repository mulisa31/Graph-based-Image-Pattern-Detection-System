import Gui.UserInterface;
import Implementations.AppLogic;
import javafx.application.Application;
import javafx.scene.image.Image;
import javafx.stage.Stage;

public class ImageGraphApp extends Application {
    
	@Override
    public void start(Stage primaryStage) {
        // Initialize the app logic layer
        AppLogic appLogic = new AppLogic();
     /*   Image icon=new Image("icon.jpg");
        primaryStage.getIcons().add(icon);*/
        
        // Initialize the GUI with the logic layer
        UserInterface mainUI = new UserInterface(primaryStage, appLogic);
        mainUI.initialize();
        
        // Ensure proper shutdown when window closes
        primaryStage.setOnCloseRequest(e -> {
            appLogic.shutdown();
          
        });
    }

    public static void main(String[] args) {
        // Launch the JavaFX application
        launch(args);
    }
}