#include <iostream>
#include <vector>
#include <algorithm> // For std::sort
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <fstream>

// Insertion Sort Implementation
void insertionSort(std::vector<int>& array, int start, int end) {
    for (int i = start + 1; i <= end; i++) {
        int key = array[i];
        int j = i - 1;
        while (j >= start && array[j] > key) {
            array[j + 1] = array[j];
            j--;
        }
        array[j + 1] = key;
    }
}

// Merge Function Implementation
void merge(std::vector<int>& array, int start, int mid, int end) {
    int leftSize = mid - start + 1;
    int rightSize = end - mid;

    std::vector<int> leftArray(leftSize);
    std::vector<int> rightArray(rightSize);

    for (int i = 0; i < leftSize; i++) {
        leftArray[i] = array[start + i];
    }
    for (int j = 0; j < rightSize; j++) {
        rightArray[j] = array[mid + 1 + j];
    }

    int i = 0, j = 0, k = start;
    while (i < leftSize && j < rightSize) {
        if (leftArray[i] <= rightArray[j]) {
            array[k++] = leftArray[i++];
        }
        else {
            array[k++] = rightArray[j++];
        }
    }

    while (i < leftSize) {
        array[k++] = leftArray[i++];
    }
    while (j < rightSize) {
        array[k++] = rightArray[j++];
    }
}

// Hybrid Sort Implementation
void hybridSort(std::vector<int>& array, int start, int end, int K) {
    int n = end - start + 1;
    if (n <= K) {
        insertionSort(array, start, end);
    }
    else {
        int mid = start + (end - start) / 2;
        hybridSort(array, start, mid, K);
        hybridSort(array, mid + 1, end, K);
        merge(array, start, mid, end);
    }
}

// Generate Sorted Array
void generateSortedArray(std::vector<int>& array) {
    int value = 0;
    for (auto& element : array) {
        element = value++;
    }
}

// Test Correctness Function
bool testCorrectness(const std::vector<int>& original, const std::vector<int>& sorted) {
    std::vector<int> expected = original;
    std::sort(expected.begin(), expected.end());
    return sorted == expected;
}

// Print Array
void printArray(const std::vector<int>& array) {
    for (const auto& element : array) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
}

int main() {
    const int numTests = 100;  // Number of tests to run
    const int arraySize = 1000; // Size of each array
    const int K = 10; // Threshold for Hybrid Sort to switch to Insertion Sort

    // Part 1: Test Correctness on Sorted Arrays
    bool allPassed = true;

    for (int i = 0; i < numTests; i++) {
        std::vector<int> array(arraySize); // Generate an array of size `arraySize`
        generateSortedArray(array); // Populate the array with sorted numbers

        std::vector<int> hybridArray = array; // Copy the array to be sorted
        hybridSort(hybridArray, 0, arraySize - 1, K); // Perform Hybrid Sort

        if (i == 0) {
            // Print input and output for the first test
            std::cout << "Test 1 Input Array (Sorted): ";
            printArray(array);
            std::cout << "Test 1 Sorted Array (Hybrid Sort): ";
            printArray(hybridArray);
        }

        std::cout << "Test " << i + 1 << ": ";
        if (testCorrectness(array, hybridArray)) {
            std::cout << "Passed" << std::endl;
        }
        else {
            std::cout << "Failed" << std::endl;
            allPassed = false;
            break; // Stop further tests if one fails
        }
    }

    if (allPassed) {
        std::cout << "All tests passed successfully on sorted arrays!" << std::endl;
    }
    else {
        std::cout << "Some tests failed on sorted arrays." << std::endl;
        return 1; // Exit early if tests fail
    }

    // Part 2: Performance Testing on Sorted Arrays and Writing to File
    std::vector<int> nValues = { 100, 200, 300, 1000, 5000, 10000 }; // Array sizes to test
    int maxK = 100; // Maximum value of K
    int numTrials = 20; // Number of trials to average over

    std::ofstream outputFile("output_sorted.csv");
    if (!outputFile.is_open()) {
        std::cerr << "Error: Could not open file for writing." << std::endl;
        return 1;
    }

    outputFile << "n,K,AverageTime\n";

    for (int n : nValues) {
        for (int K = 1; K <= maxK; K++) {
            double totalTime = 0.0;

            for (int trial = 0; trial < numTrials; trial++) {
                std::vector<int> array(n);
                generateSortedArray(array);

                auto start = std::chrono::high_resolution_clock::now();
                hybridSort(array, 0, n - 1, K);
                auto end = std::chrono::high_resolution_clock::now();

                std::chrono::duration<double> elapsedTime = end - start;
                totalTime += elapsedTime.count();
            }

            double avgTime = totalTime / numTrials;
            outputFile << n << "," << K << "," << avgTime << "\n";
        }
    }

    outputFile.close();
    std::cout << "Performance data on sorted arrays written to output_sorted.csv" << std::endl;

    return 0;
}
