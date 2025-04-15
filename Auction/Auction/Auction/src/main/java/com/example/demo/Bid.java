package com.example.demo;

/**
 * Represents a bid in an auction system.
 * This class stores information about the bid amount and the bidder.
 */
public class Bid {
    // The amount of the bid.
    private double amount;

    // The user who made the bid.
    private User bidder;

    /**
     * Constructor for creating a new Bid.
     * Initializes the bid with the specified amount and bidder.
     * @param amount The amount of the bid.
     * @param bidder The user who is making the bid.
     */
    public Bid(double amount, User bidder) {
        this.amount = amount;
        this.bidder = bidder;
    }

    /**
     * Retrieves the amount of the bid.
     * This method returns the monetary value of the bid.
     * @return The bid amount.
     */
    public double getAmount() {
        return amount;
    }

    /**
     * Retrieves the user who made the bid.
     * This method returns the bidder's information.
     * @return The bidder.
     */
    public User getBidder() {
        return bidder;
    }
}