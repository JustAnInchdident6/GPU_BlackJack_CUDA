#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <algorithm> // For std::min
#include <iostream>  // For std::cerr, std::endl (used in error checking)

// CUDA Error Checking Macro
#define CUDA_CHECK(call)                                                               \
    do {                                                                               \
        cudaError_t err = call;                                                        \
        if (err != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA Error in %s at line %d: %s (%d)\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err), err);                 \
            exit(EXIT_FAILURE);                                                        \
        }                                                                              \
    } while (0)

// Game constants
constexpr int ACE = 11;
constexpr int TEN = 10; // For Jack, Queen, King
constexpr int DEALER_STAND_LIMIT = 17;
constexpr int BLACKJACK = 21;

// CUDA device function to draw a card
__device__ __forceinline__ int draw_card(curandState *state) {
    // Generates a card value: 2-10 are their face value. J,Q,K are 10. Ace is 11 initially.
    int v = curand(state) % 13 + 2; // Card value from 2 to 14 (Ace represented as 14 initially)
    if (v >= 11 && v <= 13) return TEN; // J, Q, K
    if (v == 14) return ACE;            // Ace
    return v;                           // 2-10
}

// CUDA Kernel for running simulations
__global__ void run_simulation_kernel(
    int dealer_first_card_val,        // Value of dealer's initial visible card
    int player_first_card_val,        // Value of player's initial card
    unsigned long long total_simulations_to_run,
    unsigned long long *global_results // [0]=player_wins, [1]=player_losses, [2]=draws
) {
    unsigned int thread_id_global = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_threads_in_grid = gridDim.x * blockDim.x;

    curandState rng_state;
    curand_init(1234 + thread_id_global, 0, 0, &rng_state); // Seed per thread with offset

    unsigned long long local_wins = 0;
    unsigned long long local_losses = 0;
    unsigned long long local_draws = 0;

    // Grid-stride loop: each thread processes multiple simulations
    for (unsigned long long sim_idx = thread_id_global; sim_idx < total_simulations_to_run; sim_idx += total_threads_in_grid) {
        // --- Start of one Blackjack simulation ---

        // Player's hand
        int player_total = player_first_card_val;
        int player_aces = (player_first_card_val == ACE);

        // Player draws second card
        int player_card_2 = draw_card(&rng_state);
        player_total += player_card_2;
        if (player_card_2 == ACE) {
            player_aces++;
        }

        // Adjust for Player's Aces if busted (convert Ace from 11 to 1)
        while (player_total > BLACKJACK && player_aces > 0) {
            player_total -= 10;
            player_aces--;
        }

        // Player busts with initial two cards (player strategy: stand on two cards)
        if (player_total > BLACKJACK) {
            local_losses++;
            continue; // Go to the next simulation for this thread
        }

        // Dealer's hand
        int dealer_total = dealer_first_card_val;
        int dealer_aces = (dealer_first_card_val == ACE);

        // Dealer draws second card (hole card)
        int dealer_card_2 = draw_card(&rng_state);
        dealer_total += dealer_card_2;
        if (dealer_card_2 == ACE) {
            dealer_aces++;
        }
        // Adjust for Dealer's Aces if busted (initial two cards)
        while (dealer_total > BLACKJACK && dealer_aces > 0) {
            dealer_total -= 10;
            dealer_aces--;
        }
        // Note: If dealer has Blackjack with 2 cards, game might end, but typical simulation lets player play first.
        // Here, we assume player has played (stood on 2 cards).

        // Dealer's turn: Dealer draws cards until total is DEALER_STAND_LIMIT or more (if not busted on 2 cards)
        if (dealer_total <= BLACKJACK) { // Only if dealer didn't bust with 2 cards
            while (dealer_total < DEALER_STAND_LIMIT) {
                int dealer_drawn_card = draw_card(&rng_state);
                dealer_total += dealer_drawn_card;
                if (dealer_drawn_card == ACE) {
                    dealer_aces++;
                }
                // Adjust for Dealer's Aces if busted
                while (dealer_total > BLACKJACK && dealer_aces > 0) {
                    dealer_total -= 10;
                    dealer_aces--;
                }
            }
        }

        // Determine outcome
        if (dealer_total > BLACKJACK) { // Dealer busts
            local_wins++;
        } else if (player_total > dealer_total) { // Player has higher score
            local_wins++;
        } else if (dealer_total > player_total) { // Dealer has higher score
            local_losses++;
        } else { // player_total == dealer_total (Push/Draw)
            local_draws++;
        }
        // --- End of one Blackjack simulation ---
    }

    // Reduction of results within the block using shared memory
    __shared__ unsigned long long block_shared_results[3]; // 0: wins, 1: losses, 2: draws

    if (threadIdx.x == 0) {
        block_shared_results[0] = 0;
        block_shared_results[1] = 0;
        block_shared_results[2] = 0;
    }
    __syncthreads(); // Ensure shared memory is initialized before atomics

    atomicAdd(&block_shared_results[0], local_wins);
    atomicAdd(&block_shared_results[1], local_losses);
    atomicAdd(&block_shared_results[2], local_draws);
    __syncthreads(); // Ensure all threads in block have updated shared memory

    // First thread in block writes block results to global memory
    if (threadIdx.x == 0) {
        atomicAdd(&global_results[0], block_shared_results[0]);
        atomicAdd(&global_results[1], block_shared_results[1]);
        atomicAdd(&global_results[2], block_shared_results[2]);
    }
}

// Host function to set up and run CUDA simulations
extern "C" void run_simulation(int dealer_input_card, int player_input_card, unsigned long long num_total_simulations) {
    if (num_total_simulations == 0) {
        printf("Number of simulations cannot be zero.\n");
        return;
    }

    // Map input '1' to ACE value (11) for card values
    int dealer_start_card_val = (dealer_input_card == 1) ? ACE : dealer_input_card;
    int player_start_card_val = (player_input_card == 1) ? ACE : player_input_card;

    unsigned long long *results_d; // Device memory for results [wins, losses, draws]
    unsigned long long results_h[3] = {0, 0, 0}; // Host memory for results, initialized

    // Allocate memory on the GPU for results
    CUDA_CHECK(cudaMalloc(&results_d, 3 * sizeof(unsigned long long)));
    // Initialize results memory on GPU to zero
    CUDA_CHECK(cudaMemset(results_d, 0, 3 * sizeof(unsigned long long)));

    // Kernel launch configuration
    int threadsPerBlock = 256; // Common choice, can be tuned (e.g., 128, 256, 512)

    unsigned long long num_blocks_calculated = (num_total_simulations + threadsPerBlock - 1) / threadsPerBlock;
    if (num_blocks_calculated == 0 && num_total_simulations > 0) { // Ensure at least one block if simulations > 0
        num_blocks_calculated = 1;
    }

    // Cap num_blocks to a practical maximum for a 1D grid if necessary
    // Max gridDim.x is (2^31 - 1). A very large number of blocks might not be optimal.
    unsigned int gridDimX = static_cast<unsigned int>(num_blocks_calculated);
    const unsigned int maxGridDimX = 65535 * 2; // A large but somewhat practical cap for many scenarios. Adjust if needed.
                                             // True device max is (1<<31)-1 but launching that many blocks is unusual.
    if (num_blocks_calculated > maxGridDimX) {
        gridDimX = maxGridDimX;
        printf("Warning: Number of calculated blocks (%llu) exceeds cap (%u). Capping to %u blocks.\n",
               num_blocks_calculated, maxGridDimX, gridDimX);
        printf("The grid-stride loop in the kernel will ensure all simulations are still processed.\n");
    }
     if (gridDimX == 0 && num_total_simulations > 0) gridDimX = 1;


    dim3 gridDim(gridDimX);
    dim3 blockDim(threadsPerBlock);

    if (gridDim.x > 0) { // Only launch if there are blocks to launch
        printf("Launching kernel with %u blocks and %d threads per block to process %llu simulations.\n",
               gridDim.x, blockDim.x, num_total_simulations);
    } else {
        printf("No simulations to run or zero blocks calculated. Kernel will not be launched.\n");
    }


    // CUDA events for timing
    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    // Record start event
    CUDA_CHECK(cudaEventRecord(start_event, 0));

    // Launch the kernel only if there are simulations and blocks
    if (num_total_simulations > 0 && gridDim.x > 0) {
        run_simulation_kernel<<<gridDim, blockDim>>>(dealer_start_card_val, player_start_card_val, num_total_simulations, results_d);
        // Check for kernel launch errors immediately
        CUDA_CHECK(cudaGetLastError());
    }

    // Record stop event and synchronize
    CUDA_CHECK(cudaEventRecord(stop_event, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_event)); // Wait for all GPU work to complete

    // Copy results from GPU (device) to CPU (host)
    CUDA_CHECK(cudaMemcpy(results_h, results_d, 3 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    // Calculate elapsed time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
    double seconds = milliseconds / 1000.0;
    double simulations_per_second = 0.0;
    if (seconds > 1e-6 && num_total_simulations > 0) { // Avoid division by zero or tiny numbers
        simulations_per_second = static_cast<double>(num_total_simulations) / seconds;
    }


    // Print results
    printf("\n--- Results for %llu simulations ---\n", num_total_simulations);
    if (num_total_simulations > 0) {
        printf("Player Wins : %llu (%.2f%%)\n", results_h[0], (static_cast<double>(results_h[0]) * 100.0) / num_total_simulations);
        printf("Player Loses: %llu (%.2f%%)\n", results_h[1], (static_cast<double>(results_h[1]) * 100.0) / num_total_simulations);
        printf("Draws       : %llu (%.2f%%)\n", results_h[2], (static_cast<double>(results_h[2]) * 100.0) / num_total_simulations);
    } else {
        printf("Player Wins : 0 (0.00%%)\n");
        printf("Player Losses: 0 (0.00%%)\n");
        printf("Draws       : 0 (0.00%%)\n");
    }
    printf("\nTotal time: %.3f seconds\n", seconds);
    printf("Simulations per second: %.0f\n", simulations_per_second);

    // Free GPU memory and destroy events
    CUDA_CHECK(cudaFree(results_d));
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
}

// Main function (CPU host code) - Entry point of the program
int main() {
    int dealer_card_input;
    int player_card_input;
    unsigned long long num_sims_input;

    printf("Dealer's up card (2-10, or 1 for Ace): ");
    if (scanf("%d", &dealer_card_input) != 1) {
        fprintf(stderr, "Invalid input for dealer card.\n");
        return 1;
    }
    if (!((dealer_card_input >= 1 && dealer_card_input <= 10))) { // Basic validation
        fprintf(stderr, "Dealer card input out of expected range (1-10).\n");
        return 1;
    }


    printf("Player's first card (2-10, or 1 for Ace): ");
    if (scanf("%d", &player_card_input) != 1) {
        fprintf(stderr, "Invalid input for player card.\n");
        return 1;
    }
    if (!((player_card_input >= 1 && player_card_input <= 10))) { // Basic validation
        fprintf(stderr, "Player card input out of expected range (1-10).\n");
        return 1;
    }

    printf("Number of simulations: ");
    if (scanf("%llu", &num_sims_input) != 1) {
        fprintf(stderr, "Invalid input for number of simulations.\n");
        return 1;
    }

    printf("\nSimulating with Dealer's up-card: %d, Player's first card: %d for %llu games.\n",
           dealer_card_input, player_card_input, num_sims_input);
    printf("(Note: If you entered 1 for Ace, it will be treated as 11 in game logic.)\n\n");

    run_simulation(dealer_card_input, player_card_input, num_sims_input);

    return 0;
}