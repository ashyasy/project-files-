//package myJavaCode;

/**@Purpose The purpose of this program is to implement a dynamic programming solution to compute the maximum enjoyment points for a Knapsack Problem
 * @date November 24, 2024
 */
public class Knapsack_Problem {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		 // Define weights, values, and the weight limit
        int[] weights_forExpereince = {8, 7, 6, 5, 4};
        int[] values_forExpereince = {1500, 1600, 1700, 1800, 3000};
        int Total_weight_Limit = 20;

        // Call the function to compute the maximum enjoyment points
        int Total_Enjoyment_Points = knapsack(weights_forExpereince, values_forExpereince, Total_weight_Limit);
        System.out.println("\nThe total enjoyment point for this travel will be: " + Total_Enjoyment_Points +" points.\n\n");
    }

    public static int knapsack(int[] weight, int[] value_points, int total_weight_Limit) {
        int num = weight.length;

        // Creating a DP table to store maximum values for each weight limit
        int[][] dp = new int[num + 1][total_weight_Limit + 1];

        // Filling in the DP table
        for (int i = 1; i <= num; i++) {
            for (int j = 0; j <= total_weight_Limit; j++) {
                if (weight[i - 1] <= j) {
                    // Option 1: Exclude the item
                    // Option 2: Include the item (add its value and reduce the weight)
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i - 1][j - weight[i - 1]] + value_points[i - 1]);
                } else {
                    // If the item's weight is more than the current weight limit, exclude it
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }

        // Here, we backtrack to find which items were included
        int weights = total_weight_Limit;
        System.out.print("\nBest experience to choose: ");
        for (int i = num; i > 0; i--) {
            if (dp[i][weights] != dp[i - 1][weights]) {
                System.out.print(i + " "); // Experience index (1-based)
                weights -= weight[i - 1];
            }
        }
        System.out.println();

        // This line of code return the maximum number of enjoyment points
        return dp[num][total_weight_Limit];

	}

}
