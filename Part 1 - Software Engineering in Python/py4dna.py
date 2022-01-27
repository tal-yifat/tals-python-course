import random

def deal_cards(n_players, n_rounds):
    all_cards = [i + 1 for i in range(n_players * n_rounds)]
    random.shuffle(all_cards)
    dealt_cards = [all_cards[player * n_rounds : (player + 1) * n_rounds] \
                     for player in range(n_players)]
    print('The Dealt Cards: \n')
    for i in range(n_players): 
        print('Player', i + 1, ':', dealt_cards[i])

    return dealt_cards

def play_rounds(n_players, n_rounds, dealt_cards):
    points = [0] * n_players
    print()
    for i in range(n_rounds):
        round_cards = []
        for j in range(n_players):
            round_cards += [dealt_cards[j][i]]
        round_winner = round_cards.index(max(round_cards)) + 1
        points[round_winner - 1] += 1
        print('Round', i + 1, ':', round_cards, '- the winner is player', round_winner)

    print ('\nFinal score:', points)

def play_card_game(n_players, n_rounds):
    dealt_cards = deal_cards(n_players, n_rounds) 
    play_rounds(n_players, n_rounds, dealt_cards)
