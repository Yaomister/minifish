from chess import Chess
from evaluator import Evaluator
from accumulator import Accumulator


def __main__():
    """run the game loop."""
    print("Chess engine starting")
    evaluator = Evaluator()
    accumulator = evaluator.build_accumulator()
    game = Chess(accumulator)


    color = input("Choose color (B/W)\n") == "W"

    while not len(game.get_all_avaliable_moves()) == 0:
        print(game)
        print("")
        if game.white_moves == color:
            while True:
                try:
                    print("Your move")
                    start = input("Choose a starting square (e.g C5):\n").capitalize()
                    end = input("Choose a end square (e.g A1):\n").capitalize()
                    promotion = input("Choose piece to promote to (press enter/return if not applicable):\n").capitalize()

                    start = game.convert_coordinates(start)
                    end = game.convert_coordinates(end)

                    if (game.board[start] and Chess.is_reachable(game.board, start, abs(start[0] - end[0]), abs(start[1] - end[1])) and game.make_move(start, end, promotion)):
                        break

                    raise ValueError()
                except:
                    print("Invalid move")
        else:
            start, end, promotion = evaluator.get_best_move(game, False, 5)
            game.make_move(start, end, promotion)
            print(Chess.convert_coordinates(start) + " to " + Chess.convert_coordinates(end))

            if promotion:
                print(f"Promoted to {promotion}")

    
__main__()
        