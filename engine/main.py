from chess import Chess


if __name__ == "__main__":
    """run the game loop."""
    print("Chess engine starting.")
    game = Chess()
    color = input("Choose color (B/W)\n") == "W"


    print("Enter \"quit\" any time to quit.")
    while not len(game.get_all_avaliable_moves()) == 0:
        print(game)
        print("")
        if game.white_moves == color:
            print("Your move")
        else:
            print("Something")
        