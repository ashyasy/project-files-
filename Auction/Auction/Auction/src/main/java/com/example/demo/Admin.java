package com.example.demo;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

/**
 * Admin class responsible for managing products, users, and auctions in the system.
 */
public class Admin {
    private String username; // Admin's username
    private String password; // Admin's password (should be handled securely in a real application)
    private EnglishAuction currentEnglishAuction; // Currently active English Auction
    private SealedAuction currentSealedAuction; // Currently active Sealed Auction
    private List<Product> products = new ArrayList<>(); // List of products in the system
    private List<User> users = new ArrayList<>(); // List of users in the system
    private List<Auction> auctions = new ArrayList<>(); // List of auctions in the system

    /**
     * Constructor for Admin.
     * @param username Admin's username.
     * @param password Admin's password.
     */
    public Admin(String username, String password) {
        this.username = username;
        this.password = password;
    }

    // Methods for Product Management

    /**
     * Adds a product to the system.
     * @param id The product ID.
     * @param name The product name.
     * @param description The product description.
     * @param price The product price.
     */
    public void addProduct(String id, String name, String description, double price) {
        Product product = new Product(id, name, description, price);
        products.add(product);
        System.out.println("Product added: " + product.getName());
    }

    /**
     * Removes a product from the system by ID.
     * @param productId The ID of the product to remove.
     * @return true if the product was removed, false otherwise.
     */
    public boolean removeProduct(String productId) {
        Product productToRemove = findProductById(productId);
        if (productToRemove != null) {
            products.remove(productToRemove);
            System.out.println("Product removed: " + productToRemove.getName());
            return true;
        } else {
            System.out.println("Product not found with ID: " + productId);
            return false;
        }
    }

    /**
     * Finds a product by its ID.
     * @param productId The ID of the product to find.
     * @return The product if found, null otherwise.
     */
    private Product findProductById(String productId) {
        for (Product product : products) {
            if (product.getId().equals(productId)) {
                return product; // Found the product with the specified ID
            }
        }
        return null; // Product with the specified ID not found
    }

    // Getter methods

    public List<Product> getProducts() { return products; }
    public List<User> getUsers() { return users; }
    public List<Auction> getAuctions() { return auctions; }

    // Methods for User Management

    /**
     * Adds a user to the system.
     * @param user The user to add.
     */
    public void addUser(User user) {
        users.add(user);
        System.out.println("User added: " + user.getName());
    }

    /**
     * Removes a user from the system.
     * @param user The user to remove.
     */
    public void removeUser(User user) {
        if (users.remove(user)) {
            System.out.println("User removed: " + user.getName());
        } else {
            System.out.println("User not found: " + user.getName());
        }
    }

    // Methods for Auction Management

    /**
     * Checks if the current English auction is active.
     * @return true if the English auction is active, false otherwise.
     */
    public boolean isEnglishAuctionActive() {
        return currentEnglishAuction != null &&
                LocalDateTime.now().isAfter(currentEnglishAuction.getStartTime()) &&
                LocalDateTime.now().isBefore(currentEnglishAuction.getEndTime());
    }

    /**
     * Sets the current English auction.
     * @param auction The English auction to set.
     * @param product The product for the auction.
     */
    public void setCurrentEnglishAuction(EnglishAuction auction, Product product) {
        this.currentEnglishAuction = auction;
        auctions.add(auction);
        System.out.println("Auction set up for product: " + product.getName());
    }

    /**
     * Checks if the current Sealed auction is active.
     * @return true if the Sealed auction is active, false otherwise.
     */
    public boolean isSealedAuctionActive() {
        return currentSealedAuction != null &&
                LocalDateTime.now().isAfter(currentSealedAuction.getStartTime()) &&
                LocalDateTime.now().isBefore(currentSealedAuction.getEndTime());
    }

    /**
     * Sets the current Sealed auction.
     * @param auction The Sealed auction to set.
     * @param product The product for the auction.
     */
    public void setCurrentSealedAuction(SealedAuction auction, Product product) {
        this.currentSealedAuction = auction;
        auctions.add(auction);
        System.out.println("Auction set up for product: " + product.getName());
    }

    // Getter methods for the current auctions
    public EnglishAuction getCurrentEnglishAuction() { return currentEnglishAuction; }
    public SealedAuction getCurrentSealedAuction() { return currentSealedAuction; }
}