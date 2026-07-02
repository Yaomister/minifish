from chess import Chess
from evaluator import Evaluator
import traceback
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
        if game.turn == color:
            while True:
                try:
                    print("Your move")
                    start = input("Choose a starting square (e.g C5):\n").capitalize()
                    if start == "Quit":
                        return
                    end = input("Choose a end square (e.g A1):\n").capitalize()
                    promotion = input("Choose piece to promote to (press enter/return if not applicable):\n").capitalize()

                    start = game.convert_coordinates(start)
                    end = game.convert_coordinates(end)

                    if game.board[start] and Chess.is_reachable(game.board, start, end[0] - start[0], end[1] - start[1]) and game.make_move(start, end, promotion):
                        break
                    raise ValueError()
                except:
                    traceback.print_exc()
                    print("Invalid move")
        else:
            print("Your opponent's move")
            start, end, promotion = evaluator.get_best_move(game, False, 5)
            game.make_move(start, end, promotion)
            print(Chess.convert_coordinates(start) + " to " + Chess.convert_coordinates(end))

            if promotion:
                print(f"Promoted to {promotion}")

__main__()
        