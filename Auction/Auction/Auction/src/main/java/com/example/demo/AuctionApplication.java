package com.example.demo;

import javafx.application.Application;
import javafx.collections.FXCollections;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.stage.Stage;
import java.time.LocalDateTime;

public class AuctionApplication extends Application {
    private  Admin admin; // Admin instance to manage products, users, auctions

    private Stage primaryStage;
    private Scene mainScene;
    private VBox mainLayout;
    private Button backToHomeButton;
    private User currentUser; // The user currently placing a bid
    private TextField bidField; // Field for entering bid
    private Label currentBidLabel;
    private Button placeBidButton;




    public AuctionApplication() {
        admin = new Admin("adminUsername", "adminPassword"); // Initialize the admin with default credentials
    }

    @Override
    public void start(Stage stage) {
        this.primaryStage = stage;
        primaryStage.setTitle("Auction System");

        backToHomeButton = new Button("Back to Home");
        backToHomeButton.setOnAction(e -> primaryStage.setScene(mainScene));

        createMainScene();
        primaryStage.setScene(mainScene);
        primaryStage.show();
    }

    private void createMainScene() {
        Button engAuctionButton = new Button("English Auction");
        engAuctionButton.setOnAction(e -> {
            if (admin.isEnglishAuctionActive()) {
                System.out.println("English Auction is now active.");
                openEnglishAuctionBiddingScene(admin.getCurrentEnglishAuction());
            } else {
                showAlert("No English Auction is currently running.");
            }
        });

        Button sealAuctionButton = new Button("Sealed Auction");
        sealAuctionButton.setOnAction(e -> {
            // Assuming there is a method in 'admin' to check and get the current sealed auction
            if (admin.isSealedAuctionActive()) {
                System.out.println("Sealed Auction is now active.");
                openSealedAuctionBiddingScene(admin.getCurrentSealedAuction());
            } else {
                showAlert("No Sealed Auction is currently running.");
            }
        });
        Button viewAuctionResultsButton = new Button("View Auction Results");
        viewAuctionResultsButton.setOnAction(e -> viewAuctionResults());


        Button showProductsButton = new Button("Show All Products");
        showProductsButton.setOnAction(e -> showAllProducts());

        Button adminLoginButton = new Button("Admin Login");
        adminLoginButton.setOnAction(e -> openAdminLoginScene());





        HBox auctionTypeBox = new HBox(10, engAuctionButton, sealAuctionButton);


        Button addProductButton = new Button("Add Product");


        mainLayout = new VBox(10, auctionTypeBox, adminLoginButton, showProductsButton);
        mainScene = new Scene(mainLayout, 800, 600);
    }
    private void setCurrentUser(User user) {
        this.currentUser = user;
        // Enable the bid field and place bid button if they are initially disabled
        bidField.setDisable(false);

        placeBidButton.setDisable(false);
    }

    private void openAdminLoginScene() {
        VBox layout = new VBox(10);
        TextField usernameField = new TextField();
        usernameField.setPromptText("Enter Username");
        PasswordField passwordField = new PasswordField();
        passwordField.setPromptText("Enter Password");

        Button loginButton = new Button("Login");
        loginButton.setOnAction(e -> {
            if (adminLogin(usernameField.getText(), passwordField.getText())) {
                openAdminControlsScene();
            } else {
                System.out.println("Invalid credentials");
            }
        });

        layout.getChildren().addAll(new Label("Admin Login"), usernameField, passwordField, loginButton, backToHomeButton);
        primaryStage.setScene(new Scene(layout, 800, 600));
    }






    private void showAllProducts() {
        StringBuilder productListBuilder = new StringBuilder();
        for (Product product : admin.getProducts()) {
            productListBuilder.append(product.toString()).append("\n");
        }

        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle("Product List");
        alert.setHeaderText("All Products");

        TextArea textArea = new TextArea(productListBuilder.toString());
        textArea.setEditable(false);
        textArea.setWrapText(true);
        textArea.setMaxWidth(Double.MAX_VALUE);
        textArea.setMaxHeight(Double.MAX_VALUE);

        GridPane.setVgrow(textArea, Priority.ALWAYS);
        GridPane.setHgrow(textArea, Priority.ALWAYS);

        GridPane expContent = new GridPane();
        expContent.setMaxWidth(Double.MAX_VALUE);
        expContent.add(textArea, 0, 0);

        alert.getDialogPane().setContent(expContent);
        alert.showAndWait();
    }

    private void openAdminControlsScene() {
        VBox layout = new VBox(10);

        Button addProductButton = new Button("Add Product");
        addProductButton.setOnAction(e -> openAddProductScene());

        Button deleteProductButton = new Button("Delete Product");
        deleteProductButton.setOnAction(e -> openDeleteProductScene());

        Button startAuctionButton = new Button("Start Auction");
        startAuctionButton.setOnAction(e -> openStartAuctionScene());

        layout.getChildren().addAll(addProductButton, deleteProductButton, startAuctionButton, backToHomeButton);
        primaryStage.setScene(new Scene(layout, 800, 600));
    }

    private void openAddProductScene() {
        VBox layout = new VBox(10);
        TextField productIdField = new TextField();
        productIdField.setPromptText("Enter Product ID");
        TextField productNameField = new TextField();
        productNameField.setPromptText("Enter Product Name");
        TextField productDescriptionField = new TextField();
        productDescriptionField.setPromptText("Enter Product Description");
        TextField productPriceField = new TextField();
        productPriceField.setPromptText("Enter Product Price");

        Button submitButton = new Button("Add Product");
        submitButton.setOnAction(e -> {
            try {
                String id = productIdField.getText();
                String name = productNameField.getText();
                String description = productDescriptionField.getText();
                double price = Double.parseDouble(productPriceField.getText());

                admin.addProduct(id, name, description, price); // Add product using Admin class

                productIdField.clear();
                productNameField.clear();
                productDescriptionField.clear();
                productPriceField.clear();

                System.out.println("Product added: " + name);
            } catch (NumberFormatException ex) {
                System.out.println("Invalid price. Please enter a valid number.");
            }
        });

        layout.getChildren().addAll(
                new Label("Add New Product"),
                productIdField,
                productNameField,
                productDescriptionField,
                productPriceField,
                submitButton,
                backToHomeButton
        );

        primaryStage.setScene(new Scene(layout, 800, 600));
    }

    private void openStartAuctionScene() {
        VBox layout = new VBox(10);

        // Dropdown to select product
        ComboBox<Product> productComboBox = new ComboBox<>();
        productComboBox.setItems(FXCollections.observableArrayList(admin.getProducts()));
        productComboBox.setPromptText("Select a Product");

        // Dropdown to select auction type
        ComboBox<String> auctionTypeComboBox = new ComboBox<>();
        auctionTypeComboBox.setItems(FXCollections.observableArrayList("English Auction", "Sealed Auction"));
        auctionTypeComboBox.setPromptText("Select Auction Type");

        // Field for auction duration
        TextField durationField = new TextField();
        durationField.setPromptText("Enter Auction Duration (in minutes)");

        // Field for minimum bid
        TextField minBidField = new TextField();
        minBidField.setPromptText("Enter Minimum Bid");

        Button confirmStartButton = new Button("Start Auction");
        confirmStartButton.setOnAction(e -> {
            try {
                Product selectedProduct = productComboBox.getValue();
                String auctionType = auctionTypeComboBox.getValue();
                int duration = Integer.parseInt(durationField.getText());
                double minBid = Double.parseDouble(minBidField.getText());

                LocalDateTime startTime = LocalDateTime.now();
                LocalDateTime endTime = startTime.plusMinutes(duration);

                Auction newAuction;
                if ("English Auction".equals(auctionType)) {
                    newAuction = new EnglishAuction(selectedProduct, startTime, endTime, minBid);
                    admin.setCurrentEnglishAuction((EnglishAuction) newAuction, selectedProduct);
                } else if ("Sealed Auction".equals(auctionType)) {
                    newAuction = new SealedAuction(selectedProduct, startTime, endTime, minBid);
                    admin.setCurrentSealedAuction((SealedAuction) newAuction, selectedProduct);
                }
                    else {
                    throw new IllegalArgumentException("Invalid auction type selected.");
                }



                System.out.println("Auction started for product: " + selectedProduct.getName() + " as " + auctionType);

                productComboBox.setValue(null);
                auctionTypeComboBox.setValue(null);
                durationField.clear();
                minBidField.clear();
            } catch (NumberFormatException ex) {
                System.out.println("Invalid input. Please check your duration and bid values.");
            } catch (IllegalArgumentException ex) {
                System.out.println(ex.getMessage());
            }
        });

        layout.getChildren().addAll(
                new Label("Start Auction"),
                productComboBox,
                auctionTypeComboBox,
                durationField,
                minBidField,
                confirmStartButton,
                backToHomeButton
        );

        primaryStage.setScene(new Scene(layout, 800, 600));
    }

    private void openDeleteProductScene() {
        VBox layout = new VBox(10);
        layout.setPadding(new Insets(15, 12, 15, 12));
        layout.setSpacing(10);

        TextField productIdField = new TextField();
        productIdField.setPromptText("Enter Product ID to Delete");

        Button deleteButton = new Button("Delete Product");
        deleteButton.setOnAction(e -> {
            String productId = productIdField.getText();
            boolean isDeleted = admin.removeProduct(productId); // removeProduct should return true if deletion is successful

            if (isDeleted) {
                System.out.println("Product deleted: " + productId);
                showAlert("Product deleted successfully.");
            } else {
                System.out.println("Product not found: " + productId);
                showAlert("Product not found.");
            }

            productIdField.clear();
        });

        layout.getChildren().addAll(new Label("Delete Product"), productIdField, deleteButton, backToHomeButton);
        primaryStage.setScene(new Scene(layout, 800, 600));
    }
    private void openEnglishAuctionBiddingScene(EnglishAuction auction) {
        VBox layout = new VBox(10);
        layout.setPadding(new Insets(15, 12, 15, 12));
        layout.setSpacing(10);

        Button ashButton = new Button("Ash");
        Button shrutiButton = new Button("Shruti");
        Button sohamButton = new Button("Soham");

        ashButton.setOnAction(e -> setCurrentUser(new User("Ash")));
        shrutiButton.setOnAction(e -> setCurrentUser(new User("Shruti")));
        sohamButton.setOnAction(e -> setCurrentUser(new User("Soham")));

        bidField = new TextField();
        bidField.setPromptText("Enter your bid amount");
        bidField.setDisable(false);



        placeBidButton = new Button("Place Bid");
        placeBidButton.setOnAction(e -> placeEnglishAuctionBid(auction, bidField));
        placeBidButton.setDisable(true); // Initially disabled


        currentBidLabel = new Label("Current Highest Bid: $" + auction.getCurrentBid());

        layout.getChildren().addAll(ashButton, shrutiButton, sohamButton, currentBidLabel, bidField, placeBidButton, backToHomeButton);
        primaryStage.setScene(new Scene(layout, 800, 600));
    }


    private void showAlert(String message) {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setContentText(message);
        alert.showAndWait();
    }
    private void openBiddingScene(Auction auction) {
        VBox layout = new VBox(10);
        layout.setPadding(new Insets(15, 12, 15, 12));
        layout.setSpacing(10);

        Label infoLabel = new Label();
        TextField bidAmountField = new TextField();
        bidAmountField.setPromptText("Enter your bid amount");
        Button placeBidButton = new Button("Place Bid");

        if (auction instanceof EnglishAuction) {
            EnglishAuction englishAuction = (EnglishAuction) auction;
            infoLabel.setText("Current Highest Bid: $" + englishAuction.getCurrentBid());

            placeBidButton.setOnAction(e -> placeEnglishAuctionBid(englishAuction, bidAmountField));
        } else if (auction instanceof SealedAuction) {
            SealedAuction sealedAuction = (SealedAuction) auction;
            infoLabel.setText("This is a Sealed Auction. Place your best bid.");

            placeBidButton.setOnAction(e -> placeSealedAuctionBid(sealedAuction, bidAmountField));
        }

        layout.getChildren().addAll(infoLabel, bidAmountField, placeBidButton, backToHomeButton);
        primaryStage.setScene(new Scene(layout, 800, 600));
    }

    private void placeEnglishAuctionBid(EnglishAuction auction, TextField bidField) {
        try {
            double bidAmount = Double.parseDouble(bidField.getText());
            Bid newBid = new Bid(bidAmount, currentUser);
            auction.placeBid(newBid);

            currentBidLabel.setText("Current Bid: $" + auction.getCurrentBid());
            showAlert("Bid placed successfully. Current highest bid: $" + auction.getCurrentBid());
            if (admin.getCurrentEnglishAuction() != null && !admin.isEnglishAuctionActive()) {
                EnglishAuction englishAuction = admin.getCurrentEnglishAuction();
                englishAuction.determineAndSetWinningBid();  // Calculate the winning bid
            }
        } catch (NumberFormatException ex) {
            showAlert("Invalid bid amount. Please enter a valid number.");
        } catch (InvalidBidException ex) {
            showAlert(ex.getMessage());
        } finally {
            bidField.clear();
        }
    }

    private void openSealedAuctionBiddingScene(SealedAuction auction) {
        VBox layout = new VBox(10);
        layout.setPadding(new Insets(15, 12, 15, 12));
        layout.setSpacing(10);

        // User buttons
        Button ashButton = new Button("Ash");
        Button shrutiButton = new Button("Shruti");
        Button sohamButton = new Button("Soham");

        ashButton.setOnAction(e -> currentUser=(new User("Ash")));
        shrutiButton.setOnAction(e -> currentUser=(new User("Shruti")));
        sohamButton.setOnAction(e -> currentUser=(new User("Soham")));

        // Bid field
        PasswordField bidField = new PasswordField();
        bidField.setPromptText("Enter your bid amount");
        bidField.setDisable(false);

        // Place bid button
        placeBidButton = new Button("Place Bid");
        placeBidButton.setOnAction(e -> placeSealedAuctionBid(auction, bidField));
        placeBidButton.setDisable(false);

        // Add components to layout
        layout.getChildren().addAll(ashButton, shrutiButton, sohamButton, bidField, placeBidButton, backToHomeButton);

        primaryStage.setScene(new Scene(layout, 800, 600));
    }

    private void placeSealedAuctionBid(SealedAuction auction, TextField bidField) {
        try {
            double bidAmount = Double.parseDouble(bidField.getText());
            if (bidAmount < 0) {
                showAlert("Bid amount cannot be negative.");
                return;
            }

            Bid newBid = new Bid(bidAmount, currentUser);
            auction.submitBid(newBid);

            showAlert("Bid submitted successfully by "+ currentUser);
        } catch (NumberFormatException | InvalidBidException ex) {
            showAlert("Invalid bid amount. Please enter a valid number.");
        } finally {
            bidField.clear();
        }
    }
    private void viewAuctionResults() {
        StringBuilder results = new StringBuilder();

        if (admin.getCurrentEnglishAuction() != null) {
            // Assuming you have a way to determine if the English auction has ended
            // and to get the winning bid
            if (admin.getCurrentEnglishAuction() != null && !admin.isEnglishAuctionActive()) {
                EnglishAuction englishAuction = admin.getCurrentEnglishAuction();
                Bid winningBid = englishAuction.getWinningBid();  // Retrieve the winning bid

                if (winningBid != null) {
                    results.append("English Auction Result:\n")
                            .append("Product: ").append(englishAuction.getProduct().getName()).append("\n")
                            .append("Winning Bid: $").append(winningBid.getAmount())
                            .append(" by ").append(winningBid.getBidder().getName()).append("\n\n");
                }
            }
        }

        if (admin.getCurrentSealedAuction() != null) {
            // Similar logic for Sealed Auction
            SealedAuction sealedAuction = admin.getCurrentSealedAuction();
            if (!admin.isSealedAuctionActive()) { // Assuming the auction has ended
                Bid winningBid = sealedAuction.getWinningBid(); // Or however you retrieve the winning bid
                results.append("Sealed Auction Result:\n")
                        .append("Product: ").append(sealedAuction.getProduct().getName()).append("\n")
                        .append("Winning Bid: $").append(winningBid.getAmount())
                        .append(" by ").append(winningBid.getBidder().getName()).append("\n\n");
            }
        }

        if (results.length() == 0) {
            results.append("No finished auctions to display.");
        }

        showAlert("Auction Results", results.toString());
    }

    private void showAlert(String title, String message) {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
    }


    private boolean adminLogin(String username, String password) {
        return username.equals("admin") && password.equals("admin");
    }

    public static void main(String[] args) {
        launch(args);
    }
}