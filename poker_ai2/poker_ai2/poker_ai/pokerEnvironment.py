import random
import csv
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Define the card values and suits
CARD_VALUES = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
CARD_VALUE_RANKS = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                    '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
SUITS = ['Hearts', 'Diamonds', 'Clubs', 'Spades']

def create_model(input_size, output_size):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_size,)),
        Dense(32, activation='relu'),
        Dense(output_size, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

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
        print(f"{self.name} bets {amount} chips. Current bet: {self.current_bet}, Remaining chips: {self.chips}")

    def fold(self):
        self.folded = True
        print(f"{self.name} folds.")

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

    def make_decision(self, game_state):
        """AI decision-making logic. To be implemented in AIPlayer subclass."""
        pass

class AIPlayer(Player):
    def __init__(self, name, chips=1000, model=None):
        super().__init__(name, chips, is_ai=True)
        self.model = model  # Neural network model

    def make_decision(self, game_state):
        """
        Use the neural network model to decide on an action.
        """
        if not self.model:
            return "Fold"  # Default action if model is not available

        # Prepare input features
        input_features = self.extract_features(game_state)
        input_tensor = tf.convert_to_tensor([input_features], dtype=tf.float32)

        # Predict probabilities for each action
        predictions = self.model.predict(input_tensor, verbose=0)
        action_probabilities = predictions[0]

        # Select the action with the highest probability
        action_index = tf.argmax(action_probabilities).numpy()
        actions = ["Check", "Call", "Raise", "Fold"]
        action = actions[action_index]

        print(f"{self.name} (AI) predicts action probabilities: {action_probabilities}")
        print(f"{self.name} (AI) chooses to {action}.")

        return action

    def extract_features(self, game_state):
        """
        Extract and normalize features from the game state for the model.
        """
        # Hand strength is already evaluated
        hand_rank, _, _ = PokerHandEvaluator.evaluate_hand(self.hand + game_state.community_cards)

        # Normalize features (optional but recommended)
        normalized_hand_rank = hand_rank / 10  # Assuming max rank is 10
        normalized_player_chips = self.chips / 1000  # Assuming max chips is 1000
        normalized_current_bet = self.current_bet / 100  # Assuming max bet is 100
        normalized_min_bet = game_state.min_bet / 100  # Assuming max min_bet is 100
        normalized_pot = game_state.pot / 1000  # Assuming max pot is 1000
        num_community_cards = len(game_state.community_cards)
        normalized_num_community_cards = num_community_cards / 5  # Max 5 community cards

        return [
            normalized_hand_rank,
            normalized_player_chips,
            normalized_current_bet,
            normalized_min_bet,
            normalized_pot,
            normalized_num_community_cards
        ]

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
                    return (9, [sf_high], "Royal Flush")
                else:
                    return (9, [sf_high], "Straight Flush")
        # Four of a Kind
        if rank_counts_sorted[0][1] == 4:
            four_rank = rank_counts_sorted[0][0]
            kicker = max([rank for rank in ranks if rank != four_rank])
            return (8, [four_rank, kicker], "Four of a Kind")
        # Full House
        if rank_counts_sorted[0][1] == 3 and rank_counts_sorted[1][1] >= 2:
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
        if rank_counts_sorted[0][1] == 3:
            three_rank = rank_counts_sorted[0][0]
            kickers = sorted(set([rank for rank in sorted_ranks if rank != three_rank]), reverse=True)[:2]
            return (4, [three_rank] + kickers, "Three of a Kind")
        # Two Pair
        if rank_counts_sorted[0][1] == 2 and rank_counts_sorted[1][1] == 2:
            high_pair = rank_counts_sorted[0][0]
            low_pair = rank_counts_sorted[1][0]
            kicker = max([rank for rank in sorted_ranks if rank != high_pair and rank != low_pair])
            return (3, [high_pair, low_pair, kicker], "Two Pair")
        # One Pair
        if rank_counts_sorted[0][1] == 2:
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
    
class RuleBasedAI(Player):
    def make_decision(self, game_state):
        """
        Simple rule-based decision-making:
        - Raise if hand strength is Full House or better.
        - Call if hand strength is Three of a Kind or Two Pair.
        - Fold otherwise.
        """
        hand_rank, _, _ = PokerHandEvaluator.evaluate_hand(self.hand + game_state.community_cards)
        
        if hand_rank >= 7:  # Full House or better
            return "Raise"
        elif hand_rank >= 3:  # Two Pair or Three of a Kind
            return "Call"
        else:
            return "Fold"


class PokerEnvironment:
    def __init__(self, player_names, ai_players=[]):
        """
        Initialize the Poker Environment.

        :param player_names: List of player names.
        :param ai_players: List of player names who are AI.
        """
        self.deck = Deck()
        self.players = []
        self.pot = 0
        self.community_cards = []
        self.current_player_index = 0
        self.root = tk.Tk()
        self.root.title("Poker Game")
        self.round_number = 0
        self.big_blind_amount = 20
        self.small_blind_amount = 10
        self.big_blind_index = 1  # Will be updated after first blinds
        self.small_blind_index = 0  # Will be updated after first blinds
        self.last_raise_amount = self.big_blind_amount  # Track the last raise amount
        self.min_bet = self.big_blind_amount
        self.new_round = True

        # Create and train the model
        self.model = create_model(input_size=6, output_size=4)
        self.train_model()

        # Initialize players after the model is trained
        for name in player_names:
            if name in ai_players:
                ai_player = AIPlayer(name, model=self.model)
                self.players.append(ai_player)
            else:
                self.players.append(Player(name))

        # Setup the UI after players are initialized
        self.setup_ui()

    def train_model(self):
        """
        Train the neural network model with dummy data.
        """
        # Generate dummy training data
        # In a real scenario, replace this with actual game data
        num_samples = 10000
        X = np.random.rand(num_samples, 6)  # 6 input features
        y = np.random.randint(0, 4, size=(num_samples,))  # 4 actions

        # One-hot encode the labels
        y_encoded = tf.keras.utils.to_categorical(y, num_classes=4)

        # Train the model
        print("Training the AI model with dummy data...")
        self.model.fit(X, y_encoded, epochs=10, batch_size=32, verbose=1)
        print("Training completed.")

    def setup_ui(self):
        # Configure the grid to expand dynamically
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Main Frame to hold everything
        main_frame = tk.Frame(self.root)
        main_frame.grid(sticky="nsew")
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # Community Cards in the center
        self.community_cards_frame = tk.Frame(main_frame)
        self.community_cards_frame.grid(row=1, column=1, pady=20, sticky="n")
        self.reset_button = tk.Button(self.community_cards_frame, text="Reset Game", command=self.reset_game)
        self.reset_button.pack(pady=5)
        self.pot_label = tk.Label(self.community_cards_frame, text=f"Pot: {self.pot}")
        self.pot_label.pack()
        self.community_cards_label = tk.Label(self.community_cards_frame, text="Community Cards:")
        self.community_cards_label.pack()
        community_cards_inner_frame = tk.Frame(self.community_cards_frame)
        community_cards_inner_frame.pack()
        self.community_cards_images = []
        for _ in range(5):  # Max 5 community cards
            community_card_label = tk.Label(community_cards_inner_frame)
            community_card_label.pack(side=tk.LEFT, padx=5)
            self.community_cards_images.append(community_card_label)

        # Winner Label
        self.winner_label = tk.Label(self.community_cards_frame, text="", font=("Arial", 12, "bold"))
        self.winner_label.pack(pady=10)

        # Player Frames around the community cards
        self.player_frames = []
        positions = [(0, 0), (2, 2)]  # Positions for two players: Top-Left and Bottom-Right
        for idx, player in enumerate(self.players):
            frame = tk.Frame(main_frame)
            frame.grid(row=positions[idx][0], column=positions[idx][1], pady=10, padx=10, sticky="nsew")
            label = tk.Label(frame, text=f"{player.name}: {player.chips} chips")
            label.pack()
            player.cards_labels = []
            cards_frame = tk.Frame(frame)
            cards_frame.pack()
            for _ in range(2):  # Two cards per player
                card_label = tk.Label(cards_frame)
                card_label.pack(side=tk.LEFT, padx=5)
                player.cards_labels.append(card_label)
            # Remove action buttons since there are no human players
            # Optionally, you can still display action logs or statuses
            self.player_frames.append((frame, label, player.cards_labels, {}))  # Empty dict for action_buttons

    def update_ui(self):
        for i, player in enumerate(self.players):
            frame, label, cards_labels, action_buttons = self.player_frames[i]
            blind_text = ""
            if player.is_big_blind:
                blind_text = " (Big Blind)"
            elif player.is_small_blind:
                blind_text = " (Small Blind)"
            label.config(text=f"{player.name}: {player.chips} chips{blind_text}")

            # Update player's cards
            for j in range(len(cards_labels)):
                if j < len(player.hand):
                    card = player.hand[j]
                    try:
                        card_image = ImageTk.PhotoImage(Image.open(card.image_path()).resize((100, 145)))
                        cards_labels[j].config(image=card_image)
                        cards_labels[j].image = card_image
                    except Exception as e:
                        # Handle missing image files
                        cards_labels[j].config(text=str(card))
                        cards_labels[j].image = None
                else:
                    cards_labels[j].config(image='', text='')
                    cards_labels[j].image = None

            # Since there are no human players, no need to handle action buttons

        # Update community cards and pot label
        for i, community_card_label in enumerate(self.community_cards_images):
            if i < len(self.community_cards):
                card = self.community_cards[i]
                try:
                    card_image = ImageTk.PhotoImage(Image.open(card.image_path()).resize((100, 145)))
                    community_card_label.config(image=card_image)
                    community_card_label.image = card_image
                except Exception as e:
                    # Handle missing image files
                    community_card_label.config(text=str(card))
                    community_card_label.image = None
            else:
                community_card_label.config(image='', text='')
                community_card_label.image = None
        self.pot_label.config(text=f"Pot: {self.pot}")

    def player_action(self, action, player):
        # This method is no longer needed as there are no human players
        pass

    def after_player_action(self):
        if self.is_betting_round_over():
            self.next_phase()
        else:
            self.advance_to_next_player()
            self.update_ui()
            self.handle_ai_turn()

    def handle_ai_turn(self):
        current_player = self.players[self.current_player_index]
        if current_player.is_ai and not current_player.folded:
            # Introduce a short delay for realism
            self.root.after(1000, self.execute_ai_action, current_player)

    def execute_ai_action(self, player):
        action = player.make_decision(self)
        print(f"\n{player.name} (AI) chooses to {action}.")
        try:
            if action == "Check":
                if player.current_bet == self.min_bet:
                    player.has_acted = True
                    print(f"{player.name} checks.")
                else:
                    # AI cannot check when there is a bet; must call, raise, or fold
                    action = "Call"  # Default to Call
            if action == "Fold":
                active_players = [p for p in self.players if not p.folded]
                if len(active_players) <= 1:
                    print("Game over. Only one player remains.")
                    return
                player.fold()
                player.has_acted = True
            elif action == "Raise":
                # Minimum raise must be at least the amount of the previous raise or bet
                min_raise = max(self.last_raise_amount, self.big_blind_amount)
                min_total_bet = self.min_bet + min_raise
                max_raise = player.chips + player.current_bet
                # For AI, define a raise strategy (e.g., raise by min_raise)
                amount = self.min_bet + min_raise
                if amount > max_raise:
                    amount = max_raise  # All-in if not enough chips
                raise_amount = amount - self.min_bet
                if raise_amount < min_raise:
                    # Cannot raise; default to Call
                    action = "Call"
                else:
                    player.bet(amount - player.current_bet)
                    self.pot += (amount - player.current_bet)
                    self.min_bet = amount  # Update the min_bet to the player's total bet
                    self.last_raise_amount = raise_amount  # Update last raise amount
                    player.has_acted = True
                    print(f"{player.name} raises to {player.current_bet}. Pot is now {self.pot}.")
                    # Reset other players' has_acted to False
                    for p in self.players:
                        if p != player and not p.folded:
                            p.has_acted = False
            elif action == "Call":
                amount_to_call = self.min_bet - player.current_bet
                if amount_to_call > player.chips:
                    amount_to_call = player.chips  # All-in
                if amount_to_call > 0:
                    player.bet(amount_to_call)
                    self.pot += amount_to_call
                    print(f"{player.name} calls {amount_to_call}. Pot is now {self.pot}.")
                player.has_acted = True
        except ValueError as e:
            print(f"Error: {e}")
            return

        if self.is_betting_round_over():
            self.next_phase()
        else:
            self.advance_to_next_player()
            self.update_ui()
            self.handle_ai_turn()

    def advance_to_next_player(self):
        while True:
            self.current_player_index = (self.current_player_index + 1) % len(self.players)
            player = self.players[self.current_player_index]
            if not player.folded and not player.has_acted:
                break

    def is_betting_round_over(self):
        active_players = [player for player in self.players if not player.folded]
        return all(player.has_acted for player in active_players) and all(player.current_bet == self.min_bet for player in active_players)

    def deal_hole_cards(self):
        for player in self.players:
            player.receive_card(self.deck.deal_card())
            player.receive_card(self.deck.deal_card())
            print(f"{player.name} has been dealt: {player.hand[0]} and {player.hand[1]}")
        self.assign_blinds()
        self.start_betting_round()
        self.update_ui()
        self.handle_ai_turn()  # Start AI turn if applicable

    def deal_community_cards(self, num_cards):
        print(f"\nDealing {num_cards} community card(s).")
        for _ in range(num_cards):
            card = self.deck.deal_card()
            self.community_cards.append(card)
            print(f"Community card dealt: {card}")
        self.update_ui()

    def next_phase(self):
        self.reset_bets()
        if len(self.community_cards) == 0:
            print("\n--- Flop ---")
            self.deal_community_cards(3)  # Deal Flop
        elif len(self.community_cards) == 3:
            print("\n--- Turn ---")
            self.deal_community_cards(1)  # Deal Turn
        elif len(self.community_cards) == 4:
            print("\n--- River ---")
            self.deal_community_cards(1)  # Deal River
        else:
            print("\n--- Showdown ---")
            self.determine_winner()
            return  # Exit the method to prevent starting a new betting round
        self.start_betting_round()
        self.update_ui()
        self.handle_ai_turn()

    def reset_bets(self):
        for player in self.players:
            player.current_bet = 0
            player.has_acted = False
        self.min_bet = 0
        self.last_raise_amount = 0  # Reset last raise amount
        print("\nBets have been reset for the next betting round.")

    def start_betting_round(self):
        for player in self.players:
            player.has_acted = False
        if len(self.community_cards) == 0:
            # Pre-flop betting round
            self.current_player_index = (self.big_blind_index + 1) % len(self.players)
            if self.current_player_index >= len(self.players):
                self.current_player_index = 0
        else:
            # Post-flop betting rounds
            self.current_player_index = self.small_blind_index
        # Advance to first player who has not folded
        while self.players[self.current_player_index].folded:
            self.current_player_index = (self.current_player_index + 1) % len(self.players)
        self.last_raise_amount = self.big_blind_amount  # Reset last raise amount to big blind
        self.min_bet = max(player.current_bet for player in self.players)
        print(f"\nStarting new betting round. Min bet is {self.min_bet}.")
        print(f"Current player: {self.players[self.current_player_index].name}")

    def determine_winner(self):
        print("\nDetermining the winner...")
        active_players = [player for player in self.players if not player.folded]
        best_hands = []
        for player in active_players:
            all_cards = player.hand + self.community_cards
            rank, hand, hand_name = PokerHandEvaluator.evaluate_hand(all_cards)
            best_hands.append((rank, hand, hand_name, player))
            print(f"{player.name}'s best hand: {hand_name} with high cards {hand}")

        # Sort hands from best to worst
        best_hands.sort(reverse=True, key=lambda x: (x[0], x[1]))

        # The first element is the best hand
        best_rank, best_hand, best_hand_name, best_player = best_hands[0]

        # Find all players who have the same (rank, hand)
        winners = [hand_info[3] for hand_info in best_hands if hand_info[0] == best_rank and hand_info[1] == best_hand]

        # Debugging: Print all players' hands
        print("\nAll players' hands:")
        for hand_info in best_hands:
            print(f"Player: {hand_info[3].name}, Rank: {hand_info[0]}, Hand: {hand_info[1]}")

        if len(winners) == 1:
            winner = winners[0]
            winner.chips += self.pot
            win_text = f"{winner.name} wins the pot of {self.pot} chips with a {best_hand_name}!"
            self.winner_label.config(text=win_text)
            messagebox.showinfo("Winner", win_text)
            print(win_text)
        else:
            split_pot = self.pot // len(winners)
            for winner in winners:
                winner.chips += split_pot
            winner_names = ', '.join([winner.name for winner in winners])
            tie_text = f"It's a tie between {winner_names}! They split the pot of {self.pot} chips with a {best_hand_name}."
            self.winner_label.config(text=tie_text)
            messagebox.showinfo("Tie", tie_text)
            print(tie_text)
        self.reset_game()

    def assign_blinds(self):
        # Rotate blinds
        self.small_blind_index = (self.small_blind_index + 1) % len(self.players)
        self.big_blind_index = (self.big_blind_index + 1) % len(self.players)
        print(f"\nAssigning blinds: Small Blind - {self.players[self.small_blind_index].name}, Big Blind - {self.players[self.big_blind_index].name}")

        # Assign blinds
        for i, player in enumerate(self.players):
            player.is_small_blind = (i == self.small_blind_index)
            player.is_big_blind = (i == self.big_blind_index)
            player.current_bet = 0  # Reset current bet
            if player.is_small_blind:
                try:
                    player.bet(self.small_blind_amount)
                    self.pot += self.small_blind_amount
                    print(f"{player.name} posts small blind of {self.small_blind_amount}.")
                except ValueError as e:
                    print(f"Error: {e}")
            elif player.is_big_blind:
                try:
                    player.bet(self.big_blind_amount)
                    self.pot += self.big_blind_amount
                    print(f"{player.name} posts big blind of {self.big_blind_amount}.")
                except ValueError as e:
                    print(f"Error: {e}")
        self.min_bet = self.big_blind_amount  # Set min bet to big blind after blinds are posted
        print(f"Pot after blinds: {self.pot}")

    def reset_game(self):
        # Reset the deck, players, and community cards
        print("\nResetting game for a new round.")
        self.deck = Deck()
        self.community_cards = []
        self.pot = 0
        for player in self.players:
            player.reset_hand()
        self.round_number = 0
        self.current_player_index = 0
        self.winner_label.config(text="")  # Clear the winner label
        self.assign_blinds()
        self.deal_hole_cards()
        self.update_ui()

    def play(self):
        self.deal_hole_cards()
        self.root.mainloop()

if __name__ == "__main__":
    # Define player names and specify which players are AI
    player_names = ["Bob", "Charlie"]  # Removed "Alice"
    ai_players = ["Bob", "Charlie"]  # Both are AI players
    poker_env = PokerEnvironment(player_names, ai_players)
    poker_env.play()
