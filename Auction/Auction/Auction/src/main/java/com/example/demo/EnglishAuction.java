package com.example.demo;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

/**
 * Represents an English auction, a type of auction where bids are raised successively.
 * Extends from the Auction class.
 */
public class EnglishAuction extends Auction {
    private double currentBid; // Current highest bid in the auction
    private Bid winningBid; // Winning bid after the auction ends
    private String currentbidder; // Current highest bidder
    private Product product; // Product being auctioned
    LocalDateTime startTime = LocalDateTime.now(); // Auction start time
    private List<Bid> bids = new ArrayList<>(); // List of all bids placed
    LocalDateTime endTime = startTime.plusMinutes(1); // Auction end time

    /**
     * Constructor for EnglishAuction.
     * Initializes the auction with a product, start time, end time, and minimum bid.
     * @param product The product being auctioned.
     * @param startTime The start time of the auction.
     * @param endTime The end time of the auction.
     * @param minBid The minimum bid amount for the auction.
     */
    public EnglishAuction(Product product, LocalDateTime startTime, LocalDateTime endTime, double minBid) {
        super(product, startTime, endTime, minBid);
        this.currentBid = minBid;
    }

    /**
     * Announces the start time of the auction.
     * Displays the initial bid amount.
     */
    @Override
    public void startTime() {
        System.out.println("The starting bid for the English Auction is " + currentBid);
    }

    /**
     * Announces the end time of the auction.
     * Displays the final bid amount.
     */
    @Override
    public void endTime() {
        System.out.println("The English Auction has ended. The final bid was " + currentBid);
    }

    /**
     * Places a bid in the auction.
     * Validates and updates the current highest bid.
     * @param bid The bid to be placed.
     * @throws InvalidBidException If the bid amount is less than the current highest bid.
     */
    public void placeBid(Bid bid) throws InvalidBidException {
        if (bid.getAmount() > currentBid) {
            currentBid = bid.getAmount();
            currentbidder = bid.getBidder().getName();
            System.out.println("New bid placed: " + bid.getAmount() + " by " + bid.getBidder().getName());
        } else {
            throw new InvalidBidException("Bid too low. Current bid is: " + currentBid);
        }
    }

    /**
     * Determines and sets the winning bid at the end of the auction.
     */
    public void determineAndSetWinningBid() {
        // Winning bid determination logic
    }

    // Getter for the winning bid
    public Bid getWinningBid() {
        return winningBid;
    }

    // Getter for the current highest bid
    public double getCurrentBid() {
        return currentBid;
    }

    /**
     * Announces the minimum bid for the auction.
     */
    @Override
    public void minBid() {
        System.out.println("Minimum bid for this English Auction is $" + getMinBid());
    }
}