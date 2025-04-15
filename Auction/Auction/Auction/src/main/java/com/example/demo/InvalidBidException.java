package com.example.demo;

/**
 * Custom exception class for handling invalid bid scenarios in an auction system.
 * This class extends the standard Exception class in Java.
 * It's used to signal when a bid is invalid due to reasons like a negative amount
 */
public class InvalidBidException extends Exception {

    /**
     * Constructor for InvalidBidException.
     * Initializes the exception with a specific message detailing the reason for the invalid bid.
     * @param message message that details the reason for the exception.
     */
    public InvalidBidException(String message) {
        super(message);
    }
}