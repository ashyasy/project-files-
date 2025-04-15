
import random
import openai

# Replace 'your-api-key' with your OpenAI API key
openai.api_key = 'your-api-key'

# Define card ranks and suits
SUITS = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']

# Create a deck of cards
def create_deck():
    return [f"{rank} of {suit}" for rank in RANKS for suit in SUITS]

# Shuffle and deal cards
def shuffle_and_deal(deck, num_players):
    random.shuffle(deck)
    hands = {f"Player {i+1}": [deck.pop(), deck.pop()] for i in range(num_players)}
    return hands, deck

# Evaluate hand (simple version for now)
def evaluate_hand(hand):
    return f"Hand: {hand}"

def chatgpt_decision(game_state):
    prompt = f"""
    You are an AI playing poker. Here is the current game state:
    {game_state}

    Based on the rules of Texas Hold'em, what should you do? Options: [Fold, Check, Raise, Call].
    Explain your reasoning briefly.
    """
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

def poker_game_with_ai():
    num_players = 2  # Human vs. AI
    deck = create_deck()
    hands, deck = shuffle_and_deal(deck, num_players)
    community_cards = []

    print("Your hand:", hands["Player 1"])
    print("AI hand: [Hidden]")

    for stage in ["Flop", "Turn", "River"]:
        if stage == "Flop":
            community_cards += [deck.pop() for _ in range(3)]
        else:
            community_cards.append(deck.pop())

        print(f"Community Cards ({stage}): {community_cards}")

        # AI Decision
        ai_game_state = f"AI Hand: {hands['Player 2']}, Community Cards: {community_cards}"
        ai_decision = chatgpt_decision(ai_game_state)
        print("AI Decision:", ai_decision)

        # Player Input
        player_action = input("Your move (Fold, Check, Raise, Call): ")
        if player_action.lower() == "fold":
            print("You folded. AI wins!")
            return
        elif player_action.lower() == "raise":
            print("You raised. AI needs to respond.")
            ai_response = chatgpt_decision(ai_game_state + " Player raised.")
            print("AI Response:", ai_response)

    print("Showdown!")
    print("Your hand:", hands["Player 1"])
    print("AI hand:", hands["Player 2"])

if __name__ == "__main__":
    poker_game_with_ai()
