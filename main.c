#include <stdio.h>

extern void run_simulation(int dealer, int player, unsigned long long simulations);

int main() {
    int dealer_start, player_start;
    unsigned long long simulations;

    printf("Dealer's up card (2-10, A=1): ");
    scanf("%d", &dealer_start);
    printf("Player's total (2-21): ");
    scanf("%d", &player_start);
    printf("Number of simulations: ");
    scanf("%llu", &simulations);

    run_simulation(dealer_start, player_start, simulations);

    return 0;
}
