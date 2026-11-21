"""The chess engine used to keep track of player turn and pieces."""

import numpy as np
from accumulator import Accumulator


class Chess:
    """
    The chess engine.
    """

    def __init__(self, accumulator: Accumulator, test_setup=None):
        # 8 x 8 board, filled with tuples representing (piece, color)
        self.board = np.full((8, 8), None, dtype=object)
        self.turn = True
        self.can_castle = [[True, True], [True, True]]
        # if the previous player just pushed a pawn two squares, the opposing player can do en passant
        self.en_passant = False

        # (start, end, piece played, piece taken, en passant, can castle)
        self.previous_moves = []

        self.king_locations = [(4, 0), (4, 7)]

        # set up the default pieces on the board (or load existing board)
        if test_setup:
            self.board = test_setup["board"]
            self.turn = test_setup["turn"]
            self.can_castle = test_setup["can_castle"]
            self.en_passant = test_setup["en_passant"]
            self.king_locations = [
                next(sq for sq, p in np.ndenumerate(self.board) if p == ("K", c))
                for c in (True, False)
            ]
        else:
            # white pieces
            self.board[:, 1] = [("P", True)] * 8
            self.board[1, 0] = self.board[6, 0] = ("N", True)
            self.board[0, 0] = self.board[7, 0] = ("R", True)
            self.board[2, 0] = self.board[5, 0] = ("B", True)
            self.board[3, 0] = ("Q", True)
            self.board[4, 0] = ("K", True)
            # black pieces
            self.board[:, 6] = [("P", False)] * 8
            self.board[1, 7] = self.board[6, 7] = ("N", False)
            self.board[0, 7] = self.board[7, 7] = ("R", False)
            self.board[2, 7] = self.board[5, 7] = ("B", False)
            self.board[3, 7] = ("Q", False)
            self.board[4, 7] = ("K", False)

        # keep track of possible attack and moving directions
        self.attack_directions = self._compute_possible_attack_directions()
        self.move_directions = self._compute_possible_move_directions()

        self.accumulator = accumulator
        accumulator.reset(self.board)

    @staticmethod
    def is_reachable(board, start, dist_f, dist_r):
        """
        Tests whether the given move is reachable on the board, assuming no obstacles.
        """
        piece, is_white = board[start]

        if not piece:
            return False

        if dist_f == 0 and dist_r == 0:
            return False

        match piece:
            case "P":
                if dist_f == 0:
                    if is_white:
                        # the pawn moves forward one step (or two steps if its at the starting position)
                        return dist_r == 1 or (dist_r == 2 and start[1] == 1)
                    return dist_r == -1 or (dist_r == -2 and start[1] == 6)
                return abs(dist_f) == 1 and (dist_r == (1 if is_white else -1))
            case "N":
                # moves in a L shape
                return (abs(dist_r) == 2 and abs(dist_f) == 1) or (
                    abs(dist_r) == 1 and abs(dist_f) == 2
                )
            case "B":
                # any distance in a diagonal way
                return abs(dist_r) - abs(dist_f) == 0
            case "R":
                # any distance in a single direction
                return (dist_r == 0) or (dist_f == 0)
            case "Q":
                # any distance either diagonal or via a row or column
                return (
                    (abs(dist_r) - abs(dist_f) == 0) or (dist_f == 0) or (dist_r == 0)
                )
            case "K":
                # move depends on position for King
                return (
                    start == (4, 0 if is_white else 7)
                    and abs(dist_f) == 2
                    and dist_r == 0
                ) or (abs(dist_r) <= 1 and abs(dist_f) <= 1)

    @staticmethod
    def is_in_check(board, color, attack_directions, king_position):
        """
        Checks whether the bsoard is in check state.
        """
        # check in reverse, we take the king's position and find the pieces that can attack it
        for file, rank in attack_directions[king_position[0]][king_position[1]]:
            piece_at_attacking_pos = board[file][rank]
            if piece_at_attacking_pos and piece_at_attacking_pos[1] != color:
                match piece_at_attacking_pos[0]:
                    case "Q" | "B" | "R":
                        displacement = (
                            king_position[0] - file,
                            king_position[1] - rank,
                        )
                        if displacement not in [
                            (2, 1),
                            (1, 2),
                            (2, -1),
                            (1, -2),
                            (-2, -1),
                            (-2, 1),
                            (-1, -2),
                            (-1, 2),
                        ]:
                            directions = []
                            match piece_at_attacking_pos[0]:
                                case "Q":
                                    directions += [
                                        (1, 1),
                                        (1, -1),
                                        (-1, 1),
                                        (-1, -1),
                                        (1, 0),
                                        (0, 1),
                                        (-1, 0),
                                        (0, -1),
                                    ]
                                case "R":
                                    directions += [(1, 0), (0, 1), (-1, 0), (0, -1)]
                                case "B":
                                    directions += [(1, 1), (1, -1), (-1, -1), (-1, 1)]

                            factor = max(abs(displacement[0]), abs(displacement[1]))

                            # use directions because Q, R, B can travel any amount
                            direction = (
                                int(displacement[0] / factor),
                                int(displacement[1] / factor),
                            )
                            if direction in directions:
                                # check for blockers
                                for i in range(1, 8):
                                    new_file = file + direction[0] * i
                                    new_rank = rank + direction[1] * i
                                    if new_rank in [-1, 8] or new_file in [-1, 8]:
                                        break

                                    if (new_file, new_rank) == tuple(king_position):
                                        return True
                                    if board[new_file][new_rank]:
                                        break
                    case "K" | "P" | "N":
                        targets = []
                        if piece_at_attacking_pos[0] == "N":
                            targets = [
                                (1, 2),
                                (2, 1),
                                (1, -2),
                                (-1, -2),
                                (-1, 2),
                                (2, -1),
                                (-2, -1),
                                (-2, 1),
                            ]
                        elif piece_at_attacking_pos[0] == "K":
                            targets = [
                                (1, 0),
                                (0, 1),
                                (1, 1),
                                (1, -1),
                                (-1, 1),
                                (-1, -1),
                                (-1, 0),
                                (0, -1),
                            ]
                        elif piece_at_attacking_pos[0] == "P":
                            targets = (
                                [(1, -1), (-1, -1)]
                                if color is True
                                else [(1, 1), (-1, 1)]
                            )

                        # return true if any of these pieces can reach the king
                        if (
                            king_position[0] - file,
                            king_position[1] - rank,
                        ) in targets:
                            return True
        return False

    @staticmethod
    def convert_coordinates(coordinates):
        """Convert the coordinate to alphabetic notation."""
        letters = ["A", "B", "C", "D", "E", "F", "G", "H"]

        if isinstance(coordinates, str):
            return (letters.index(coordinates[0]), int(coordinates[1]) - 1)
        if isinstance(coordinates, tuple):
            return letters[coordinates[0]] + str(coordinates[1] + 1)

        raise TypeError()

    def is_valid_move(self, start, end, promotion=None):
        """Check that the move is legal."""

        piece = self.board[start[0]][start[1]]

        # in case theres no piece or its not their turn
        if piece is None or piece[1] != self.turn:
            return False

        # check if end position is out of bounds
        if end[0] < 0 or end[0] > 7 or end[1] < 0 or end[1] > 7:
            return False

        # check if they can promote when they want to promote
        if promotion and (
            piece[0] != "P"
            or promotion not in ["Q", "R", "B", "N"]
            or end[1] not in [0, 7]
        ):
            return False

        if piece[0] == "P" and end[1] in [0, 7] and not promotion:
            return False

        # the distance they travelled
        dist_f = end[0] - start[0]
        dist_r = end[1] - start[1]

        # you cant place in a square that already has a piece of the same color
        if self.board[end] is not None and self.board[end][1] == self.turn:
            return False

        # check pawn movement validity
        if piece[0] == "P":
            if dist_f == 0:
                if self.board[end] is not None:
                    return False
            else:
                is_capturing_move = self.board[end] is not None
                # check if they make a valid capturing move or en passant
                is_en_passant = (
                    self.board[end] is None
                    and self.en_passant is not False
                    and end[0] == self.en_passant
                    and (
                        (self.turn and start[1] == 4 and end[1] == 5)
                        or (not self.turn and start[1] == 3 and end[1] == 2)
                    )
                )
                if not (is_capturing_move or is_en_passant):
                    return False

        # check if there's anything blocking the path for a rook, bishop, or queen
        if not (piece[0] == "K" and abs(dist_f) == 2):
            # bishop sliding
            if abs(dist_f) == abs(dist_r):
                factor_f = int(dist_f / abs(dist_f))
                factor_r = int(dist_r / abs(dist_r))
                for i in range(1, abs(dist_f)):
                    if (
                        self.board[start[0] + i * factor_f, start[1] + i * factor_r]
                        is not None
                    ):
                        return False
            if dist_f == 0:
                factor_r = int(dist_r / abs(dist_r))
                for i in range(1, abs(dist_r)):
                    if self.board[start[0]][start[1] + i * factor_r] is not None:
                        return False
            if dist_r == 0:
                factor_f = int(dist_f / abs(dist_f))
                for i in range(1, abs(dist_f)):
                    if self.board[start[0] + i * factor_f][start[1]] is not None:
                        return False

            check_board = self.board.copy()
            if piece[0] == "P" and dist_f != 0 and check_board[end] == None:
                check_board[end[0]][start[1]] = None

            check_board[end] = check_board[start]
            check_board[start] = None
            king_square = (
                end if piece[0] == "K" else self.king_locations[0 if self.turn else 1]
            )
            return not Chess.is_in_check(
                check_board, self.turn, self.attack_directions, king_square
            )

        # check if they can castle at all (if they've already castled before)
        if not self.can_castle[0 if self.turn else 1][0 if dist_f == 2 else 1]:
            return False

        # cannot castle with a piece in between the king and rook
        if dist_f == 2:
            for i in range(5, 7):
                if self.board[i][start[1]] is not None:
                    return False
        else:
            for i in range(1, 4):
                if self.board[i][start[1]] is not None:
                    return False

        # cannot castle out of a check
        if Chess.is_in_check(
            self.board,
            self.turn,
            self.attack_directions,
            self.king_locations[0 if self.board[start][1] else 1],
        ):
            return False

        # we play the move and see if its in check
        check_board = self.board.copy()

        if dist_f == 2:
            check_board[5, start[1]] = check_board[start]
        else:
            check_board[3, start[1]] = check_board[start]

        # cannot castle through a square that is under attack
        check_board[start] = None
        if Chess.is_in_check(
            check_board,
            self.turn,
            self.attack_directions,
            (5, start[1]) if dist_f == 2 else (3, start[1]),
        ):
            return False

        if dist_f == 2:
            check_board[6, start[1]] = self.board[start]
            check_board[5, start[1]] = check_board[7, start[1]]
            check_board[7, start[1]] = None
        else:
            check_board[2, start[1]] = self.board[start]
            check_board[3, start[1]] = check_board[0, start[1]]
            check_board[0, start[1]] = None

        # cannot castle into a check
        if Chess.is_in_check(check_board, self.turn, self.attack_directions, end):
            return False

        return True

    def make_move(self, start, end, promotion):
        """Play the move."""
        if self.is_valid_move(start, end, promotion):
            self._play_move(start, end, promotion)
            return True
        return False

    def undo_move(self):
        """Undo the last move."""
        if len(self.previous_moves) == 0:
            return

        record = self.previous_moves.pop()
        king_moved = record["king_locations"] != self.king_locations

        self.turn = record["turn"]
        self.en_passant = record["en_passant"]
        self.king_locations = record["king_locations"]
        self.can_castle = record["can_castle"]

        for square, _, _ in record["added"]:
            self.board[square] = None

        for square, letter, color in record["removed"]:
            self.board[square] = (letter, color)

        # the king square is baked into the features, so rebuild if the king moved
        if king_moved:
            self.accumulator.reset(self.board)
        else:
            self.accumulator.apply_difference(
                added=record["removed"], removed=record["added"]
            )

    def get_all_avaliable_moves(self):
        """Return all moves playable on this board."""
        moves = []
        for start, square in np.ndenumerate(self.board):
            if square and square[1] == self.turn:
                if square[0] == "P":
                    i = 5 if self.turn else 6
                else:
                    i = ["K", "Q", "R", "B", "N"].index(square[0])
                for end in self.move_directions[start[0]][start[1]][i]:
                    if square[0] == "P" and end[1] in [0, 7]:
                        for promotion in ["Q", "R", "B", "N"]:
                            if self.is_valid_move(start, end, promotion):
                                moves.append((start, end, promotion))
                    else:
                        if self.is_valid_move(start, end, None):
                            moves.append((start, end, False))
        return moves

    def __str__(self):
        """Give a texual representation of the board."""

        display = "\x1b[31m"
        display += "White's Move" if self.turn else "Black's Move"
        display += "\x1b[0m\n"
        for i in range(7, -1, -1):
            display += "\x1b[31m" + str(i + 1) + "\x1b[0m"
            for j in range(8):
                if not self.board[j, i]:
                    display += "  "
                elif self.board[j, i][1]:
                    display += f" \x1b[37m{self.board[j, i][0]}\x1b[0m"
                else:
                    display += f" \x1b[30m{self.board[j, i][0]}\x1b[0m"
            display += "\n"
        display += " "
        for i in range(8):
            display += " \x1b[31m" + "ABCDEFGH"[i] + "\x1b[0m"

        return display

    def _play_move(self, start, end, promotion):
        """Play the move."""

        added, removed = [], []

        # store previous moves for the minimax
        self.previous_moves.append(
            {
                "added": added,
                "removed": removed,
                "turn": self.turn,
                "en_passant": self.en_passant,
                "can_castle": self.can_castle,
                "king_locations": list(self.king_locations),
                "can_castle": [list(row) for row in self.can_castle],
            }
        )

        # reset en passant (file 0 is falsy, so reset unconditionally)
        self.en_passant = False

        if self.board[start][0] == "P":
            # capturing en passant
            if (start[0] - end[0]) != 0 and self.board[end] is None:
                en_passant_square = (end[0], start[1])
                if self.board[en_passant_square] is not None:
                    removed.append(
                        (
                            en_passant_square,
                            self.board[en_passant_square][0],
                            self.board[en_passant_square][1],
                        )
                    )
                    self.board[en_passant_square] = None
            if abs(start[1] - end[1]) == 2:
                self.en_passant = end[0]

        # castling
        if (self.board[start][0] == "K") and abs(end[0] - start[0]) == 2:
            castling_end = (5, start[1]) if start[0] - end[0] < 0 else (3, start[1])
            castling_start = (7, start[1]) if start[0] - end[0] < 0 else (0, start[1])

            added.append(
                (
                    castling_end,
                    self.board[castling_start][0],
                    self.board[castling_start][1],
                )
            )
            self.board[castling_end] = self.board[castling_start]
            removed.append(
                (
                    castling_start,
                    self.board[castling_start][0],
                    self.board[castling_start][1],
                )
            )
            self.board[castling_start] = None

        # revoke castling rights when the king or a rook moves, or a rook is captured
        for square in (start, end):
            for color, back_rank in ((0, 0), (1, 7)):
                if square == (4, back_rank):
                    self.can_castle[color] = [False, False]
                elif square == (7, back_rank):
                    self.can_castle[color][0] = False
                elif square == (0, back_rank):
                    self.can_castle[color][1] = False

        # remove the piece that was captured
        if self.board[end] is not None:
            removed.append((end, self.board[end][0], self.board[end][1]))

        self.board[end] = self.board[start]
        removed.append((start, self.board[start][0], self.board[start][1]))
        # empty the old position
        self.board[start] = None

        if promotion:
            self.board[end] = (promotion, self.board[end][1])

        added.append((end, self.board[end][0], self.board[end][1]))

        if self.board[end][0] == "K":
            self.king_locations[0 if self.turn else 1] = end
            self.accumulator.reset(self.board)
        else:
            self.accumulator.apply_difference(added, removed)

        self.turn = not self.turn

    def _compute_possible_attack_directions(self):
        """Precompute the possible positions each square is able to be attacked from."""

        # for each square on the board, keep track of what other squares can attack it
        # This is hypothetical, like imagine if every piece was there at the same time
        log = [[[] for _ in range(8)] for _ in range(8)]

        for file, rank in np.ndindex(self.board.shape):
            knight_jumps = [
                (1, 2),
                (2, 1),
                (-1, 2),
                (-1, -2),
                (1, -2),
                (-2, 1),
                (-2, -1),
                (2, -1),
            ]
            # log the possible places a knight can jump
            for jump in knight_jumps:
                new_pos = (file + jump[0], rank + jump[1])
                if (
                    new_pos[0] >= 0
                    and new_pos[0] <= 7
                    and new_pos[1] >= 0
                    and new_pos[1] <= 7
                ):
                    log[file][rank].append(new_pos)

            # log the possible places a rook or queen can attack
            for i in range(8):
                if i != file:
                    log[file][rank].append((i, rank))
                if i != rank:
                    log[file][rank].append((file, i))
                # log the possible places a bishop, queen, king, or pawn can attack
                if i != 0:
                    diagonals = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
                    for d in diagonals:
                        new_pos = (file + i * d[0], rank + i * d[1])
                        if (
                            new_pos[0] >= 0
                            and new_pos[0] <= 7
                            and new_pos[1] >= 0
                            and new_pos[1] <= 7
                        ):
                            log[file][rank].append(new_pos)
        return log

    def _compute_possible_move_directions(self):
        """Precompute the possible position each piece at each square is able to move to."""
        # K, Q, R, B, N, white P, black P
        dimension = len(self.board)
        log = [
            [[[] for _ in range(7)] for _ in range(dimension)] for _ in range(dimension)
        ]
        for file, rank in np.ndindex(self.board.shape):
            log[file][rank][0] = [
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
            ]
            if file == 4 and rank in [0, 7]:
                # castling
                log[file][rank][0].append((-2, 0))
                log[file][rank][0].append((2, 0))

            for i in range(1, 8):
                # queen and rook can slide across
                for j in [1, 2]:
                    log[file][rank][j].append((0, i))
                    log[file][rank][j].append((i, 0))
                    log[file][rank][j].append((0, -i))
                    log[file][rank][j].append((-i, 0))
                # queen and bishop can go diagonal
                for j in [1, 3]:
                    log[file][rank][j].append((i, i))
                    log[file][rank][j].append((i, -i))
                    log[file][rank][j].append((-i, i))
                    log[file][rank][j].append((-i, -i))

            log[file][rank][4] = [
                (1, 2),
                (-1, 2),
                (1, -2),
                (-1, -2),
                (2, 1),
                (2, -1),
                (-2, 1),
                (-2, -1),
            ]

            log[file][rank][5] = [(0, 1), (1, 1), (-1, 1)]
            if rank == 1:
                log[file][rank][5].append((0, 2))
            log[file][rank][6] = [(0, -1), (1, -1), (-1, -1)]
            if rank == 6:
                log[file][rank][6].append((0, -2))

        # remove moves that are out of bounds
        for file, rank in np.ndindex(self.board.shape):
            for i in range(7):
                to_remove = []
                for j in range(len(log[file][rank][i])):
                    move = log[file][rank][i][j]
                    new_pos = (file + move[0], rank + move[1])
                    if (
                        new_pos[0] < 0
                        or new_pos[0] > 7
                        or new_pos[1] < 0
                        or new_pos[1] > 7
                    ):
                        to_remove.append(move)
                    else:
                        log[file][rank][i][j] = new_pos
                for pos in to_remove:
                    log[file][rank][i].remove(pos)

        return log
