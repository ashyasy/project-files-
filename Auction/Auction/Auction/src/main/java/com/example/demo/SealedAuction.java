package com.example.demo;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

/**
 * Represents a sealed auction, extending the Auction class.
 * In a sealed auction, bids are submitted without other participants knowing the bid amounts.
 * This class manages the auction's timeline, submitted bids, and determines the winning bid.
 */
public class SealedAuction extends Auction {
    // List of bids submitted in the sealed auction.
    private List<Bid> sealedBids;

    // The winning bid of the auction.
    private Bid winningBid;

    /**
     * Constructor for creating a SealedAuction.
     * Initializes the auction with a product, start and end times, and a minimum bid.
     * @param product   The product being auctioned.
     * @param startTime The start time of the auction.
     * @param endTime   The end time of the auction.
     * @param minBid    The minimum bid amount for the auction.
     */
    public SealedAuction(Product product, LocalDateTime startTime, LocalDateTime endTime, double minBid) {
        super(product, startTime, endTime, minBid);
        sealedBids = new ArrayList<>();
    }

    @Override
    public void startTime() {
        System.out.println("The sealed auction has started.");
    }

    @Override
    public void endTime() {
        System.out.println("The sealed auction has ended.");
        determineWinningBid();
    }

    @Override
    public void minBid() {
        System.out.println("Minimum bid for this sealed auction is $" + getMinBid());
    }

    /**
     * Submits a bid to the sealed auction.
     * Verifies if the auction is still open before accepting the bid.
     * @param bid The bid to be submitted.
     * @throws InvalidBidException If the bid amount is negative or the auction is closed.
     */
    public void submitBid(Bid bid) throws InvalidBidException {
        if (bid.getAmount() < 0) {
            throw new InvalidBidException("Bid amount cannot be negative.");
        }

        if (!isAuctionOpen()) {
            throw new InvalidBidException("Auction is closed. Cannot submit new bids.");
        }
        sealedBids.add(bid);
        System.out.println("Bid submitted by " + bid.getBidder().getName());
    }

    /**
     * Retrieves the winning bid of the sealed auction.
     *
     * @return The winning bid.
     */
    public Bid getWinningBid() {
        return winningBid;
    }

    /**
     * Checks if the auction is currently open.
     *
     * @return true if the auction is currently open, false otherwise.
     */
    private boolean isAuctionOpen() {
        LocalDateTime currentTime = LocalDateTime.now();
        return currentTime.isAfter(getStartTime()) && currentTime.isBefore(getEndTime());
    }

    /**
     * Determines the winning bid based on the highest bid amount.
     * Iterates through all sealed bids to find the highest amount.
     */
    private void determineWinningBid() {
        if (!sealedBids.isEmpty()) {
            winningBid = sealedBids.get(0);
            for (Bid bid : sealedBids) {
                if (bid.getAmount() > winningBid.getAmount()) {
                    winningBid = bid;
                }
            }
            System.out.println("The winning bid is $" + winningBid.getAmount() +
                    " by " + winningBid.getBidder().getName());
        } else {
            System.out.println("No bids were submitted in this sealed auction.");
        }
    }
}

