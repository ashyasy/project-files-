package com.example.demo;

import java.time.LocalDateTime;

/**
 * Abstract class representing a generic auction.
 * This class defines the common properties and methods for different types of auctions.
 */
public abstract class Auction {
    private LocalDateTime startTime; // Start time of the auction
    private LocalDateTime endTime; // End time of the auction
    private double minBid; // Minimum bid amount for the auction
    private Product product; // Product being auctioned

    /**
     * Constructor for Auction with specified start time, end time, minimum bid, and product.
     * @param product   The product for the auction.
     * @param startTime The start time of the auction.
     * @param endTime   The end time of the auction.
     * @param minBid    The minimum bid for the auction.
     */
    public Auction(Product product, LocalDateTime startTime, LocalDateTime endTime, double minBid) {
        this.startTime = startTime;
        this.endTime = endTime;
        this.minBid = minBid;
        this.product = product;
    }

    /**
     * Constructor for Auction with only product specified.
     * Used when the auction does not require start/end time and minimum bid initially.
     * @param product The product for the auction.
     */
    public Auction(Product product) {
        this.product = product;
    }

    // Getter for the product
    public Product getProduct() {
        return product;
    }

    // Abstract methods to be implemented by subclasses
    public abstract void startTime();

    public abstract void endTime();

    public abstract void minBid();

    // Getter for the minimum bid
    public double getMinBid() {
        return minBid;
    }

    // Getter for the start time
    public LocalDateTime getStartTime() {
        return startTime;
    }

    // Getter for the end time
    public LocalDateTime getEndTime() {
        return endTime;
    }
}



