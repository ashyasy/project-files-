# data_collection.py

import random
import os
import csv
import logging
import numpy as np

# Configure logging to include timestamps and log levels
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("poker_simulation.log"),
        logging.StreamHandler()
    ]
)

# Define the card values and suits
CARD_VALUES = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
CARD_VALUE_RANKS = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                    '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
SUITS = ['Hearts', 'Diamonds', 'Clubs', 'Spades']

class Card:
    def __init__(self, value, suit):
        self.value = value
        self.suit = suit
        self.rank = CARD_VALUE_RANKS[value]

    def __repr__(self):
        return f"{self.value} of {self.suit}"

    def image_path(self):
        value = self.value.lower()
        suit = self.suit.lower()
        if value == 'j':
            value = 'jack'
        elif value == 'q':
            value = 'queen'
        elif value == 'k':
            value = 'king'
        elif value == 'a':
            value = 'ace'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "PNG-cards-1.3", f"{value}_of_{suit}.png")

class Deck:
    def __init__(self):
        self.cards = [Card(value, suit) for value in CARD_VALUES for suit in SUITS]
        random.shuffle(self.cards)

    def deal_card(self):
        return self.cards.pop() if self.cards else None

class Player:
    def __init__(self, name, chips=1000, is_ai=False):
        self.name = name
        self.hand = []
        self.chips = chips
        self.current_bet = 0
        self.folded = False
        self.is_big_blind = False
        self.is_small_blind = False
        self.checked = False
        self.has_acted = False
        self.is_ai = is_ai  # Flag to indicate if the player is an AI

    def bet(self, amount):
        if amount > self.chips:
            raise ValueError(f"{self.name} does not have enough chips to bet {amount}.")
        self.chips -= amount
        self.current_bet += amount
        logging.info(f"{self.name} bets {amount} chips. Current bet: {self.current_bet}, Remaining chips: {self.chips}")

    def call(self, amount):
        if amount > self.chips:
            # Player goes all-in
            self.all_in()
        else:
            self.bet(amount)
            logging.info(f"{self.name} calls {amount} chips.")

    def all_in(self):
        amount = self.chips
        self.chips = 0
        self.current_bet += amount
        logging.info(f"{self.name} goes all-in with {amount} chips. Current bet: {self.current_bet}, Remaining chips: {self.chips}")

    def fold(self):
        self.folded = True
        logging.info(f"{self.name} folds.")

    def receive_card(self, card):
        self.hand.append(card)

    def reset_bet(self):
        self.current_bet = 0
        self.checked = False

    def reset_hand(self):
        self.hand = []
        self.folded = False
        self.checked = False
        self.has_acted = False
        self.current_bet = 0

    def __repr__(self):
        return f"{self.name} with {self.chips} chips"

class RuleBasedAI(Player):
    def __init__(self, name, **kwargs):
        """
        Initialize the Rule-Based AI player.

        :param name: Player's name.
        :param kwargs: Additional arguments for the Player class.
        """
        super().__init__(name, **kwargs)

    def make_decision(self, environment, allowed_actions):
        """
        Make a decision based on predefined rules.

        :param environment: Current game environment.
        :param allowed_actions: List of actions the player can take.
        :return: Action as a string ("Check", "Call", "Raise", "Fold", "All-In").
        """
        # Evaluate hand strength
        hand_rank, _, _ = PokerHandEvaluator.evaluate_hand(self.hand + environment.community_cards)

        # Simple rule-based strategy:
        # - If hand_rank >= 7 (Full House or better), Raise
        # - If hand_rank >= 5 (Straight or better), Call
        # - If hand_rank >= 2 (One Pair), Call or Raise based on pot size
        # - Else, Fold or Check based on opponent's actions

        if "Raise" in allowed_actions and hand_rank >= 7:
            action = "Raise"
        elif "Call" in allowed_actions and hand_rank >= 5:
            action = "Call"
        elif "Call" in allowed_actions and hand_rank >= 2:
            # Decide to Raise if pot is large relative to chips
            if environment.pot > (self.chips * 0.1):
                action = "Raise"
            else:
                action = "Call"
        elif "Check" in allowed_actions:
            action = "Check"
        else:
            action = "Fold"

        # Additional rules can be added here for more sophisticated strategies

        # Log the decision
        logging.info(f"{self.name} decides to {action} with a hand rank of {hand_rank} and {self.chips} chips.")
        return action

class PokerHandEvaluator:
    @staticmethod
    def evaluate_hand(cards):
        ranks = [card.rank for card in cards]
        suits = [card.suit for card in cards]
        rank_counts = {rank: ranks.count(rank) for rank in set(ranks)}
        is_flush, flush_suit = PokerHandEvaluator.is_flush(suits)
        is_straight, straight_high = PokerHandEvaluator.is_straight(ranks)

        # Sort ranks in descending order
        sorted_ranks = sorted(ranks, reverse=True)
        rank_counts_sorted = sorted(rank_counts.items(), key=lambda x: (-x[1], -x[0]))

        # Check for Straight Flush
        if is_flush and is_straight:
            flush_ranks = [card.rank for card in cards if card.suit == flush_suit]
            is_sf, sf_high = PokerHandEvaluator.is_straight(flush_ranks)
            if is_sf:
                if sf_high == 14:
                    return (10, [sf_high], "Royal Flush")
                else:
                    return (9, [sf_high], "Straight Flush")
        # Four of a Kind
        if rank_counts_sorted and rank_counts_sorted[0][1] == 4:
            four_rank = rank_counts_sorted[0][0]
            kicker = max([rank for rank in ranks if rank != four_rank])
            return (8, [four_rank, kicker], "Four of a Kind")
        # Full House
        if rank_counts_sorted and rank_counts_sorted[0][1] == 3 and len(rank_counts_sorted) > 1 and rank_counts_sorted[1][1] >= 2:
            three_rank = rank_counts_sorted[0][0]
            two_rank = rank_counts_sorted[1][0]
            return (7, [three_rank, two_rank], "Full House")
        # Flush
        if is_flush:
            flush_cards = sorted([card.rank for card in cards if card.suit == flush_suit], reverse=True)
            return (6, flush_cards[:5], "Flush")
        # Straight
        if is_straight:
            return (5, [straight_high], "Straight")
        # Three of a Kind
        if rank_counts_sorted and rank_counts_sorted[0][1] == 3:
            three_rank = rank_counts_sorted[0][0]
            kickers = sorted(set([rank for rank in sorted_ranks if rank != three_rank]), reverse=True)[:2]
            return (4, [three_rank] + kickers, "Three of a Kind")
        # Two Pair
        if rank_counts_sorted and len(rank_counts_sorted) > 1 and rank_counts_sorted[0][1] == 2 and rank_counts_sorted[1][1] == 2:
            high_pair = rank_counts_sorted[0][0]
            low_pair = rank_counts_sorted[1][0]
            kicker = max([rank for rank in sorted_ranks if rank != high_pair and rank != low_pair])
            return (3, [high_pair, low_pair, kicker], "Two Pair")
        # One Pair
        if rank_counts_sorted and rank_counts_sorted[0][1] == 2:
            pair_rank = rank_counts_sorted[0][0]
            # Ensure unique and sorted kickers
            kickers = sorted(set([rank for rank in sorted_ranks if rank != pair_rank]), reverse=True)[:3]
            return (2, [pair_rank] + kickers, "One Pair")
        # High Card
        return (1, sorted_ranks[:5], "High Card")

    @staticmethod
    def is_flush(suits):
        for suit in set(suits):
            if suits.count(suit) >= 5:
                return True, suit
        return False, None

    @staticmethod
    def is_straight(ranks):
        unique_ranks = sorted(set(ranks), reverse=True)
        for i in range(len(unique_ranks) - 4):
            window = unique_ranks[i:i+5]
            if window[0] - window[4] == 4:
                return True, window[0]
        # Check for Wheel (Ace to Five Straight)
        if set([14, 2, 3, 4, 5]).issubset(set(ranks)):
            return True, 5
        return False, None

class PokerEnvironment:
    def __init__(self, player_names, ai_players=[], data_filename='poker_data.csv', model=None, encoder=None):
        """
        Initialize the Poker Environment.

        :param player_names: List of player names.
        :param ai_players: List of player names who are AI.
        :param data_filename: Filename to save collected data.
        :param model: Not used in Rule-Based AI.
        :param encoder: Not used in Rule-Based AI.
        """
        self.deck = Deck()
        self.players = []
        self.pot = 0
        self.community_cards = []
        self.current_player_index = 0
        self.round_number = 0
        self.big_blind_amount = 20
        self.small_blind_amount = 10
        self.big_blind_index = 1
        self.small_blind_index = 0
        self.last_raise_amount = self.big_blind_amount  # Track the last raise amount
        self.min_bet = self.big_blind_amount
        self.new_round = True

        # Initialize data collection
        self.data_filename = data_filename
        self.data = []  # List to store data entries

        # Initialize players
        for name in player_names:
            if name in ai_players:
                ai_player = RuleBasedAI(name, chips=1000, is_ai=True)
                self.players.append(ai_player)
            else:
                self.players.append(Player(name))

    def collect_data(self, player, action):
        """
        Collect features and the corresponding action taken by the player.
        """
        features = self.extract_features(player)
        action_mapping = {"Check": 0, "Call": 1, "Raise": 2, "Fold": 3, "All-In": 4}
        action_label = action_mapping.get(action, 3)  # Default to 'Fold' if action not found
        self.data.append(features + [action_label])

    def extract_features(self, player):
        """
        Extract and normalize features from the game state for the model.
        """
        hand_rank, _, _ = PokerHandEvaluator.evaluate_hand(player.hand + self.community_cards)

        normalized_hand_rank = hand_rank / 10  # Assuming max rank is 10
        normalized_player_chips = player.chips / 1000  # Assuming max chips is 1000
        normalized_current_bet = player.current_bet / 100  # Assuming max bet is 100
        normalized_min_bet = self.min_bet / 100  # Assuming max min_bet is 100
        normalized_pot = self.pot / 1000  # Assuming max pot is 1000
        num_community_cards = len(self.community_cards)
        normalized_num_community_cards = num_community_cards / 5  # Max 5 community cards

        return [
            normalized_hand_rank,
            normalized_player_chips,
            normalized_current_bet,
            normalized_min_bet,
            normalized_pot,
            normalized_num_community_cards
        ]

    def save_data_to_csv(self):
        """
        Save the collected data to a CSV file.
        """
        header = ["hand_rank", "player_chips", "current_bet", "min_bet", "pot", "num_community_cards", "action"]
        # Check if file exists to write header only once
        file_exists = os.path.isfile(self.data_filename)
        with open(self.data_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(header)
            writer.writerows(self.data)
        logging.info(f"Data saved to {self.data_filename}")
        self.data = []  # Clear data after saving

    def run_full_game(self):
        """
        Simulate a full game between AI players and collect data.
        """
        # Assign blinds
        self.assign_blinds()

        # Deal hole cards
        for player in self.players:
            player.receive_card(self.deck.deal_card())
            player.receive_card(self.deck.deal_card())
            logging.info(f"{player.name} has been dealt: {player.hand[0]} and {player.hand[1]}")

        # Start betting rounds and deal community cards
        phases = ["Pre-flop", "Flop", "Turn", "River"]
        for phase in phases:
            logging.info(f"\n--- {phase} ---")
            if phase == "Flop":
                self.deal_community_cards(3)
            elif phase in ["Turn", "River"]:
                self.deal_community_cards(1)

            self.start_betting_round()

            # Collect data after each betting round
            self.save_data_to_csv()

        # Showdown and determine winner
        self.determine_winner()

    def deal_community_cards(self, num_cards):
        logging.info(f"Dealing {num_cards} community card(s).")
        for _ in range(num_cards):
            card = self.deck.deal_card()
            if card:
                self.community_cards.append(card)
                logging.info(f"Community card dealt: {card}")
            else:
                logging.warning("No more cards in the deck to deal.")

    def assign_blinds(self):
        # Rotate blinds
        self.small_blind_index = (self.small_blind_index + 1) % len(self.players)
        self.big_blind_index = (self.big_blind_index + 1) % len(self.players)
        logging.info(f"\nAssigning blinds: Small Blind - {self.players[self.small_blind_index].name}, Big Blind - {self.players[self.big_blind_index].name}")

        # Assign blinds
        for i, player in enumerate(self.players):
            player.is_small_blind = (i == self.small_blind_index)
            player.is_big_blind = (i == self.big_blind_index)
            player.current_bet = 0  # Reset current bet
            if player.is_small_blind:
                try:
                    player.bet(self.small_blind_amount)
                    self.pot += self.small_blind_amount
                    logging.info(f"{player.name} posts small blind of {self.small_blind_amount} chips.")
                except ValueError as e:
                    logging.warning(f"Error: {e}")
            elif player.is_big_blind:
                try:
                    player.bet(self.big_blind_amount)
                    self.pot += self.big_blind_amount
                    logging.info(f"{player.name} posts big blind of {self.big_blind_amount} chips.")
                except ValueError as e:
                    logging.warning(f"Error: {e}")
        self.min_bet = self.big_blind_amount  # Set min bet to big blind after blinds are posted
        logging.info(f"Pot after blinds: {self.pot} chips")

    def start_betting_round(self, max_iterations=1000):
        """
        Start a betting round.
        :param max_iterations: Maximum number of iterations to prevent infinite loops.
        """
        # Reset player actions
        for player in self.players:
            player.has_acted = False

        # Determine betting order
        # Starting from the player next to big blind
        starting_index = (self.big_blind_index + 1) % len(self.players)
        self.current_player_index = starting_index

        # Continue betting until all players have acted and bets are equal
        betting_active = True
        iterations = 0  # To prevent infinite loops
        while betting_active and iterations < max_iterations:
            player = self.players[self.current_player_index]
            if not player.folded and player.chips > 0:
                # Determine allowed actions based on game state
                if self.min_bet == player.current_bet:
                    allowed_actions = ["Check", "Raise", "All-In"]
                else:
                    allowed_actions = ["Call", "Raise", "Fold", "All-In"]

                # AI makes decision based on allowed actions
                action = player.make_decision(self, allowed_actions)
                self.collect_data(player, action)
                self.execute_action(action, player)
            else:
                if player.folded:
                    logging.info(f"{player.name} has folded and cannot act.")
                elif player.chips == 0:
                    logging.info(f"{player.name} is all-in and cannot act.")
            self.current_player_index = (self.current_player_index + 1) % len(self.players)

            # Check if betting round is over
            betting_active = not self.is_betting_round_over()

            iterations += 1

        if iterations >= max_iterations:
            logging.warning("Maximum iterations reached in betting round. Ending round to prevent infinite loop.")

    def execute_action(self, action, player):
        """
        Execute the player's action.
        """
        try:
            if action == "Check":
                player.has_acted = True
                logging.info(f"{player.name} checks.")
            elif action == "Fold":
                player.fold()
                player.has_acted = True
            elif action == "Raise":
                # Define raise logic
                min_raise = max(self.last_raise_amount, self.big_blind_amount)
                raise_amount = min_raise
                total_raise = self.min_bet + raise_amount
                if player.chips >= raise_amount:
                    player.bet(raise_amount)
                    self.pot += raise_amount
                    self.min_bet = total_raise
                    self.last_raise_amount = raise_amount
                    player.has_acted = True
                    logging.info(f"{player.name} raises by {raise_amount} chips. Total bet: {player.current_bet}. Pot is now {self.pot} chips.")
                else:
                    # Player cannot raise the desired amount, go all-in
                    if player.chips > 0:
                        player.all_in()
                        self.pot += player.current_bet
                        self.min_bet = player.current_bet
                        self.last_raise_amount = player.current_bet
                        player.has_acted = True
                        logging.info(f"{player.name} cannot raise by {raise_amount} chips and goes all-in with {player.current_bet} chips. Pot is now {self.pot} chips.")
                    else:
                        # Player cannot raise or go all-in, must fold
                        player.fold()
                        player.has_acted = True
            elif action == "Call":
                amount_to_call = self.min_bet - player.current_bet
                if amount_to_call > player.chips:
                    if player.chips > 0:
                        player.all_in()
                        self.pot += player.current_bet
                        player.has_acted = True
                        logging.info(f"{player.name} cannot call {amount_to_call} chips and goes all-in with {player.current_bet} chips. Pot is now {self.pot} chips.")
                    else:
                        player.fold()
                        player.has_acted = True
                        logging.info(f"{player.name} cannot call {amount_to_call} chips and has no chips left, so they fold.")
                else:
                    player.call(amount_to_call)
                    self.pot += amount_to_call
                    player.has_acted = True
                    logging.info(f"{player.name} calls {amount_to_call} chips. Pot is now {self.pot} chips.")
            elif action == "All-In":
                if player.chips > 0:
                    player.all_in()
                    self.pot += player.current_bet
                    if player.current_bet > self.min_bet:
                        self.min_bet = player.current_bet
                        self.last_raise_amount = player.current_bet
                    player.has_acted = True
                    logging.info(f"{player.name} goes all-in. Pot is now {self.pot} chips.")
                else:
                    player.fold()
                    player.has_acted = True
                    logging.info(f"{player.name} cannot go all-in and folds.")
        except ValueError as e:
            logging.warning(f"Error: {e}")
            # Decide on default action, e.g., fold
            player.fold()
            player.has_acted = True

    def force_valid_action(self, player):
        """
        Force the player to take a valid action if they have chosen an invalid one.
        Priority: Call > Raise > Fold
        """
        if player.chips >= (self.min_bet - player.current_bet):
            action = "Call"
        elif player.chips > 0:
            action = "All-In"
        else:
            action = "Fold"
        logging.info(f"Force {player.name} to {action} due to invalid action choice.")
        self.collect_data(player, action)
        self.execute_action(action, player)

    def is_betting_round_over(self):
        """
        Determine if the betting round is over.
        """
        active_players = [player for player in self.players if not player.folded and player.chips > 0]
        if len(active_players) <= 1:
            return True  # Only one player remains

        # Check if all active players have acted and their bets are equal
        all_acted = all(player.has_acted for player in active_players)
        all_bets_equal = all(player.current_bet == self.min_bet for player in active_players)

        return all_acted and all_bets_equal

    def determine_winner(self):
        """
        Determine the winner(s) of the game.
        """
        logging.info("\n--- Showdown ---")
        active_players = [player for player in self.players if not player.folded]
        best_hands = []
        for player in active_players:
            all_cards = player.hand + self.community_cards
            rank, hand, hand_name = PokerHandEvaluator.evaluate_hand(all_cards)
            best_hands.append((rank, hand, hand_name, player))
            logging.info(f"{player.name}'s best hand: {hand_name} with high cards {hand}")

        # Sort hands from best to worst
        best_hands.sort(reverse=True, key=lambda x: (x[0], x[1]))

        # The first element is the best hand
        if best_hands:
            best_rank, best_hand, best_hand_name, best_player = best_hands[0]

            # Find all players who have the same (rank, hand)
            winners = [hand_info[3] for hand_info in best_hands if hand_info[0] == best_rank and hand_info[1] == best_hand]

            if len(winners) == 1:
                winner = winners[0]
                winner.chips += self.pot
                logging.info(f"{winner.name} wins the pot of {self.pot} chips with a {best_hand_name}!")
            else:
                split_pot = self.pot // len(winners)
                for winner in winners:
                    winner.chips += split_pot
                winner_names = ', '.join([winner.name for winner in winners])
                logging.info(f"It's a tie between {winner_names}! They split the pot of {self.pot} chips with a {best_hand_name}.")
        else:
            logging.info("All players folded. No winner.")

    def reset_game(self):
        """
        Reset the game state for the next round.
        """
        logging.info("\n--- Resetting Game for Next Round ---")
        self.deck = Deck()
        self.community_cards = []
        self.pot = 0
        for player in self.players:
            player.reset_hand()
        self.round_number += 1

    def play_simulation(self, num_games=1000):
        """
        Play multiple simulated games to collect data.
        """
        for game_num in range(1, num_games + 1):
            logging.info(f"\n=== Starting Game {game_num} ===")
            self.run_full_game()
            self.reset_game()
        # After simulations, save the data
        self.save_data_to_csv()

if __name__ == "__main__":
    # Define player names and specify which players are AI
    player_names = ["Bob", "Charlie"]  # Only AI players
    ai_players = ["Bob", "Charlie"]  # Both are AI players

    # Initialize the Poker environment
    poker_env = PokerEnvironment(
        player_names,
        ai_players,
        data_filename='poker_data3.csv'
    )

    # Play simulations to collect data
    poker_env.play_simulation(num_games=10)  # Start with 1 game for testing
