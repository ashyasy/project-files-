package com.example.demo;

/**
 * Represents a user in an auction system.
 * This class stores information about the user's name, address, and payment details.
 */
public class User {
    // The name of the user.
    private String name;

    // The address of the user.
    private String address;

    // The payment details of the user.
    private String paymentDetail;

    // The currently logged-in user.
    private User currentUser;

    /**
     * Constructor for creating a new User with a specified name.
     * Initializes the user's name and sets address and payment details to empty strings.
     *
     * @param name The name of the user.
     */
    public User(String name) {
        this.address = "";
        this.paymentDetail = "";
        this.name = name;
    }

    /**
     * Sets the address for the user.
     * This method updates the user's address.
     *
     * @param address The new address of the user.
     */
    public void setAddress(String address) {
        this.address = address;
    }

    /**
     * Sets the current user.
     * This private method is used to update the current user's information.
     *
     * @param user The user to set as the current user.
     */
    private void setCurrentUser(User user) {
        this.currentUser = user;
        // Other relevant code, e.g., enabling fields/buttons
    }

    /**
     * Sets the payment details for the user.
     * This method updates the user's payment information.
     *
     * @param paymentDetail The new payment details of the user.
     */
    public void setPaymentDetail(String paymentDetail) {
        this.paymentDetail = paymentDetail;
    }

    /**
     * Retrieves the name of the user.
     *
     * @return The name of the user.
     */
    public String getName() {
        return name;
    }

    /**
     * Retrieves the payment details of the user.
     *
     * @return The payment details of the user.
     */
    public String getPaymentDetail() {
        return paymentDetail;
    }

}
