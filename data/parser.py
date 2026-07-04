import os
import re
import math
import numpy as np
import zstandard as zstd
from chess import Chess
from accumulator import Accumulator
import traceback


def get_move(game: Chess, san: str):
    """
    Parse the move from algebraic notation (san).
    """

    # use regex to split up the algebraic notation
    regex = "^([NBRQK])?([a-h])?([1-8])?(x)?([a-h][1-8])(?:=([NBRQ]))?$"
    turn = game.turn
    # castling
    if san == "O-O" or san == "O-O-O":
        rank = 0 if turn else 7
        return (4, rank), (6, rank) if san == "O-O" else (2, rank), None
    
    move = re.match(regex, san)

    if not move:
        return None, None, None
    
    # the piece moved
    piece = move.group(1) or "P"
    # when the file or rank is ambiguous
    dfile  = move.group(2)
    drank = move.group(3)
    end_file, end_rank = game.convert_coordinates(move.group(5).capitalize())
    promotion = move.group(6)

    possible_starts = game.attack_directions[end_file][end_rank]

    for possible_start in possible_starts:
        square = game.board[possible_start[0]][possible_start[1]]
        if square is not None and  square[0] == piece and square[1] == turn:
            if not dfile and not drank:
                return possible_start, (end_file, end_rank), promotion
            if dfile and ["A", "B", "C", "D", "E", "F", "G", "H"].index(dfile.upper()) == possible_start[0]:
                return possible_start, (end_file, end_rank), promotion
            if drank and int(drank) - 1 == possible_start[1]:
                return possible_start, (end_file, end_rank), promotion

    return None, None, None

def get_feature_indicies(game : Chess):
    """
    Get the feature index of the board.
    """
    board = game.board

    kings_locations = game.king_locations

    features = {True: [], False : []}

    for perspective in (True, False):
        king_square = kings_locations[perspective]
        for file in range(8):
            for rank in range(8):
                square = board[file][rank]
                if not square or square[0] == 'K':
                    continue
                features[perspective].append(Accumulator.get_index(king_square,  (file, rank), square[0], square[1], perspective))
    
    # add padding because we want the tensors to stack when training
    white = features[True] + [40960] * (30 - len(features[True]))
    black = features[False] + [40960] * (30 - len(features[False]))

    return white, black
    

def process_game(current_headers, current_moves):
    """
    Process a game.
    """

    if "BOT" in current_headers.get("WhiteTitle", "") or "BOT" in current_headers.get("BlackTitle", ""):
        return [], [], [], []
    if current_headers.get("Termination", "") == "Time forfeit":
        return [], [], [], []
    if "Variant" in current_headers:
        return [], [], [], []
    
    game = Chess(accumulator = Accumulator(np.zeros((256, 40960)), np.zeros(256)))

    white_perspective = []
    black_perspective = []
    color = []
    game_scores = []
    
    just_played = False
    for token in re.finditer(r'\{([^}]*)\}|(\S+)', " ".join(current_moves)):
        comment, word = token.group(1), token.group(2)
        if comment is not None:
            if  just_played:
                move = re.search(r'\[%eval ([^\]]+)\]', comment)
                if move:
                    evaluation = move.group(1)
                    if "#" in evaluation:
                        score = -1.0 if "-" in evaluation else 1.0
                    else:
                        score = math.tanh(float(evaluation))
                    
                    white, black = get_feature_indicies(game)

                    white_perspective.append(white)
                    black_perspective.append(black)
                    color.append(game.turn)
                    game_scores.append(score)

            just_played = False
            continue
        if re.fullmatch(r'\d+\.+', word) or word in ("1-0", "0-1", "1/2-1/2", "*"):
            continue
            
        move = word.rstrip("?!")
        if not move or move.startswith("$"):
            continue

        try:
            start, end, promotion = get_move(game, move)
            if start and end:
                game.make_move(start, end, promotion)
                just_played = True
            else:
                break
        except Exception as e:
            print(e)
            traceback.print_exc()
    return game_scores, black_perspective, white_perspective, color

def save_batch(scores, black_perspective, white_perspective, colors, batch_index):
    """
    Save the batch.
    """
    path = os.path.join("./training/", f"dataset_{batch_index}.npz")
    np.savez(path, white_perspective = np.array(white_perspective, dtype=np.int32), black_perspective=np.array(black_perspective, dtype=np.int32), scores = np.array(scores, dtype=np.float32), colors = np.array(colors, dtype=np.bool))
    print(f"saved batch {batch_index}")


if __name__ == "__main__":
    batch_index = 0

    with open(os.path.join("data/lichess.zst"), "rb") as f:
        # stream it as bytes
        dctx = zstd.ZstdDecompressor()
        stream = dctx.stream_reader(f)
        buffer = b""
        number_of_games_processed = 0

        scores = []
        black_perspective = []
        white_perspective = []
        colors = []

        while True:
            current_headers = {}
            current_moves = []
            # read 64 vbutes at once
            chunked = stream.read(65536)
            if not chunked:
                break
            buffer += chunked
            lines = buffer.split(b"\n")
            buffer = lines[-1]
            # we add a buffer because 64 bytes might result in half a command at the end
            for line in lines[:-1]:
                line = line.decode("utf-8", errors="ignore").strip()
                # a move
                if line.startswith("["):
                    key = re.match(r'\[(\w+)', line).group(1)
                    value = re.search(r'"(.+)"', line).group(1)
                    current_headers[key] = value
                # end of game
                elif line == "":
                    game_scores, game_black_perspective, game_white_perspective, game_color = process_game(current_headers, current_moves)
                    black_perspective.extend(game_black_perspective)
                    white_perspective.extend(game_white_perspective)
                    scores.extend(game_scores)
                    colors.extend(game_color)
                    current_headers = {}
                    current_moves = []
                    if len(scores) > 10000000:
                        save_batch(scores, black_perspective, white_perspective, colors, batch_index)
                        batch_index += 1
                        black_perspective.clear()
                        white_perspective.clear()
                        colors.clear()
                        scores.clear()
                else:
                    current_moves.append(line)