package com.example.demo;

/**
 * Represents a product in an auction system.
 * This class stores information about a product's ID, name, description, and price.
 */
public class Product {
    // Unique identifier for the product.
    private String id;

    // Name of the product.
    private String name;

    // Description of the product.
    private String description;

    // Price of the product.
    private double price;

    /**
     * Constructor for creating a new Product.
     * Initializes the product with an ID, name, description, and price.
     * @param id The unique identifier for the product.
     * @param name The name of the product.
     * @param description A brief description of the product.
     * @param price The price of the product.
     */
    public Product(String id, String name, String description, double price) {
        this.id = id;
        this.name = name;
        this.description = description;
        this.price = price;
    }

    // Getters and setters for product properties.

    /**
     * Retrieves the ID of the product.
     * @return The product's ID.
     */
    public String getId() {
        return id;
    }

    /**
     * Sets the ID of the product.
     * @param id The new ID for the product.
     */
    public void setId(String id) {
        this.id = id;
    }

    /**
     * Retrieves the name of the product.
     * @return The product's name.
     */
    public String getName() {
        return name;
    }

    /**
     * Sets the name of the product.
     * @param name The new name for the product.
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     * Retrieves the description of the product.
     * @return The product's description.
     */
    public String getDescription() {
        return description;
    }

    /**
     * Sets the description of the product.
     * @param description The new description for the product.
     */
    public void setDescription(String description) {
        this.description = description;
    }

    /**
     * Retrieves the price of the product.
     * @return The product's price.
     */
    public double getPrice() {
        return price;
    }

    /**
     * Sets the price of the product.
     * @param price The new price for the product.
     */
    public void setPrice(double price) {
        this.price = price;
    }

    /**
     * Returns a string representation of the product.
     * @return String containing the product's details.
     */
    @Override
    public String toString() {
        return "Product{" +
                "id='" + id + '\'' +
                ", name='" + name + '\'' +
                ", description='" + description + '\'' +
                ", price=" + price +
                '}';
    }
}