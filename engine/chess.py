import numpy as np


class Chess:

    def __init__(self, accumulator , test_setup = None):
        # 8 x 8 board, filled with tuples representing (piece, color)
        self.board = np.full((8, 8), None, dtype=object)
        self.white_moves = True
        # can player castle? each value represents short-castling and long-castling for white and black
        self.can_castle  = [[True, True], [True, True]]
        # if the previous player just pushed a pawn two squares, the opposing player can do en passant
        self.en_passant = False

        # (start, end, piece played, piece taken, en passant, can castle)
        self.previous_moves = []

        if test_setup:
            self.board = test_setup["board"]
            self.white_moves = test_setup['white_moves']
            self.can_castle = test_setup['can_castle']
            self.en_passant = test_setup['en_passant']
        else:
            # white pieces 
            self.board[:, 1] = [("P", True)] * 8
            self.board[1, 0] = self.board[6, 0] = ("N", True)
            self.board[0, 0] = self.board[7, 0] = ("R", True)
            self.board[2, 0] = self.board[5, 0] = ("B", True)
            self.board[3, 0] = ("Q", True)
            self.board[4, 0] = ("K", True)
            # black pieces
            self.board[:, 6]  = [("P", False)] * 8
            self.board[1, 7] = self.board[6, 7] = ("N", False)
            self.board[0, 7] = self.board[7, 7] = ("R", False)
            self.board[2, 7] = self.board[5, 7] = ("B", False)
            self.board[3, 7] = ("Q", False)
            self.board[4, 7] = ("K", False)
        
        self.attack_directions = self._compute_possible_attack_directions()
        self.move_directions = self._compute_possible_move_directions()

        self.accumulator = accumulator

    

    

    @staticmethod
    def is_reachable(board, start, dist_f, dist_r):
        piece, is_white = board[start]

        if (dist_f == 0 and dist_r == 0):
            return False
     
        match piece:
            case "P":
                if (dist_f == 0):
                    if is_white:
                        return dist_f == 1 or (dist_f == 2 and start[1] == 1)
                    return dist_f == -1 or (dist_f == -2 and start[1] == 6)
                return abs(dist_f) ==1 and (dist_r == 1 if is_white else -1)
            case "K":
                return (abs(dist_r) == 2 and abs(dist_f) == 1) or (abs(dist_r) == 1 and abs(dist_f) == 2)
            case "B":
                return abs(dist_r) - abs(dist_f) == 0
            case "R":
                return (dist_r == 0) or (dist_f == 0)
            case "Q":
                return (abs(dist_r) - abs(dist_f) == 0) or (dist_f == 0) or (dist_r == 0)
            case "K":
                return (start == (4, 0 if is_white else 7) and abs(dist_f) == 2 and dist_r == 0) or (abs(dist_r) <= 1 and abs(dist_f) <= 2)
            
    @staticmethod
    def is_in_check(board, color, attack_directions, king_position):
        for (file, rank) in attack_directions[king_position[0]][king_position[1]]:
            piece_at_attacking_pos = board[file][rank]
            if piece_at_attacking_pos[1] != color:
                match piece_at_attacking_pos[0]:
                    case "Q" | "B" | "R":
                        displacement = (king_position[0] - file, king_position[1] - rank)
                        if displacement not in [(2, 1), (1, 2), (2, -1), (1, -2), (-2, -1), (-2, 1), (-1, -2), (-1, 2)]:
                            directions = []
                            match piece_at_attacking_pos[0]:
                                case "Q":
                                    directions += [(1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (0, 1)]
                                case "R":
                                    directions += [(1, 0), (0, 1), (-1, 0), (0, -1)]
                                case "B":
                                    directions += [(1, 1), (1, -1), (-1, -1), (-1, 1)]

                            factor = max(abs(displacement[0]), abs(displacement[1]))
                            direction = (int(displacement[0]/ factor), int(displacement[1]/ factor))
                            if direction in displacement:
                                # check for blockers
                                for i in range(1, 8):
                                    new_file = file + direction[0] * i
                                    new_rank = rank + direction[1] * i
                                    if new_rank in [-1, 8] or new_file in [-1, 8]:
                                        break
                                    
                                    if board[new_file][new_rank]:
                                        if board[new_file][new_rank][0] == "K":
                                            return True
                                        break
                        return False
                    case "K" | "P" | "N":
                        targets = []
                        if piece_at_attacking_pos[0] == "N":
                            targets = [(1, 2), (2, 1), (1, -2), (-1, -2), (-1, 2), (2, -1), (-2, -1), (-2, 1)]
                        elif piece_at_attacking_pos[0] == "K":
                            targets = [(1, 0), (0, 1), (1, 1), (1, -1), (-1, 1), (-1, -1), (-1, 0), (0, -1)]
                        elif (piece_at_attacking_pos[0] == "P"):
                            targets = [(1, 1), (-1, 1)] if color is True else [(1, -1), (-1, -1)]

                        if (king_position[0] - file, king_position[1] - rank) in targets:
                            return True
        return False
    
    def is_valid_move(self, start, end, promotion = None):
        piece = self.board[start]

        # check if end position is out of bounds
        if (end[0] < 0 or end[0] > 7 or end[1] < 0 or end[1] > 7):
            return False
        
        # check if they can promote
        if promotion and (piece != "P" or promotion not in ['Q', 'K', "R", "B"] or end[1] not in [0, 7]):
            return False
        if piece == "P" and end[1] in [0, 7] and not promotion:
            return False
        
        dist_f = end[0] - start[0]
        dist_r = end[1] - start[1]
        # check if they can castle
        if not self.can_castle[0 if self.white_moves else 1][0 if dist_f == 2 else 1]:
            return False
        
        # check if there's anything blocking the way of a rook or bishop
        return True
    
    def make_move(self, start, end, promotion):
        if (self.is_valid_move(start, end, promotion)):
            self._play_move(start, end, promotion)
        return
    

    def convert_coordinates(self, coordinates):
        letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
        if isinstance(coordinates, str):
            return (letters.index(coordinates[0]), int(coordinates[1]) - 1)
        if isinstance(coordinates, tuple):
            return letters[coordinates[0] - 1] + str(coordinates[1] - 1)
        
        raise TypeError()


    def _play_move(self, start, end, promotion):

        added, removed = [], []

        # reset enpassant
        if self.en_passant:
            self.en_passant = False

        if self.board[start][0] == "P":
            # capturing en passant
            if (start[0] - end[0]) != 0 and self.board[end][0] == None:
                en_passant_square = (end[0], start[1])
                removed.append((en_passant_square, self.board[en_passant_square][0], self.board[en_passant_square][1]))
                self.board[en_passant_square] = None
            if (abs(start[1] - end[1]) == 2):
                self.en_passant = end[0]
            
        # castling
        if (self.board[start][0] == 'K') and abs(end[0] - start[0]) == 2:

            # revoke castling rights
            self.can_castle[self.white_moves] = [False, False]

            castling_end = (5, start[1]) if start[0] - end[0] < 0 else (0, start[1])
            castling_start = (7, start[1]) if start[0] - end[0] < 0 else (3, start[1])

            added.append((castling_end, self.board[castling_start][0], self.board[castling_start][1]))
            self.board[castling_end] = self.board[castling_start]
            removed.append((castling_start, self.board[castling_start][0], self.board[castling_start][1]))
            self.board[castling_start] = None
        
        
        self.board[end] = self.board[start]
        removed.append((start, self.board[start][0], self.board[start][1]))
        self.board[start] = None

        if (promotion):
            self.board[end][0] = promotion

        added.append((end, self.board[end][0], self.board[end][1]))

        self.accumulator.apply_difference(added, removed)
        
            
    
    def get_all_avaliable_moves(self):
        moves = []
        for start, square in np.ndenumerate(self.board):
            if square and square[1] == self.white_moves:
                if square[0] == "P":
                    i = 5 if self.white_moves else 6
                else:
                    i = ["K", "Q", "B", "R", "N"].index(square[0])

            for end in self.move_directions[start[0]][start[1]][i]:
                if (end[1] in [0, 7]):
                    for promotion in ["Q", "R", "B", "K"]:
                        if (self.is_valid_move(start, end, promotion)):
                            moves.append((start, end, promotion))
                else:
                    if (self.is_valid_move(start, end , None)):
                        moves.append((start, end))
                    
        return moves



    def __str__(self):
        display = "\x1b[31m"
        display += "White's Move" if self.white_moves else "Black's Move"
        display += "\x1b[0m\n"
        for i in range(7, -1, -1):
            display +=  "\x1b[31m" + str(i + 1) + "\x1b[0m"
            for j in range(8):
                if not self.board[j, i]:
                    display += " "
                elif self.board[j, i][1]:
                    display += f" \x1b[37m{self.board[j, i][0]}\x1b[0m"
                else:
                    display += f" \x1b[30m{self.board[j, i][0]}\x1b[0m"
            display += "\n"
        display += " "
        for i in range(8):
            display += " \x1b[31m" + "ABCDEFGH"[i] + "\x1b[0m"

        return display
        

    # precompute possible positions to be attacked from
    def _compute_possible_attack_directions(self):
        log = [[[] for _ in range(8)] for _ in range(8)]
        for file, rank in np.ndindex(self.board.shape):
            knight_jumps = [(1, 2), (2, 1), (-1, 2), (-1, -2), (1, -2), (-2, 1), (-2, -1), (2, -1)]
            # log the possible places a knight can jump
            for jump in knight_jumps:
                new_pos = [file + jump[0], rank + jump[1]]
                if (new_pos[0] >= 0 and new_pos[0] <= 7 and new_pos[1] >= 0 and new_pos[0] <= 7 ):
                    log[file][rank].append(new_pos)

            # log the possible places a rook or queen can attack
            for i in range(8):
                if (i != file):
                    log[file][rank].append([i, rank])
                if (i != rank):
                    log[file][rank].append([file, i])
                # log the possible places a bishop, queen, king, or pawn can attack
                if (i != 0):
                    diagonals = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
                    for d in diagonals:
                        new_pos = [file + i * d[0], rank + i * d[1]]
                        if (new_pos[0] >= 0 and new_pos[0] <= 7 and new_pos[1] >= 0 and new_pos[0] <= 7 ):
                            log[file][rank].append(new_pos)
        return log
    
    # precomplute possible positions it can move to 
    def _compute_possible_move_directions(self):
        # K, Q, R, B, N, white P, black P
        dimension = len(self.board)
        log = [[[[] for _ in range(7)] for _ in range(dimension)] for _ in range(dimension)]
        for file, rank in np.ndindex(self.board.shape):
            log[file][rank][0] = [(1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)]
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
                    log[file][rank][j].append((-i, -i))
                    log[file][rank][j].append((-i, -i))

            log[file][rank][4] = [(1, 2), (-1, 2), (1, -2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)]

            if (rank > 0):
                log[file][rank][5] = [(0, 1), (1, 1), (-1, 1)]
            if (rank == 0):
                log[file][rank][5].append((0, 2))
            if (rank < 6):
                log[file][rank][6] = [(0, -1), (1, -1), (-1, -1)]
            if (rank == 6):
                log[file][rank][6].append((0, -2))

        # remove moves that are out of bounds
        for file, rank in np.ndindex(self.board.shape):
            for i in range(6):
                to_remove = []
                for move in log[file][rank][i]:
                    new_pos = (file + move[0], rank + move[1])
                    if new_pos[0] < 0 or new_pos[0] > 7 or new_pos[1] < 0 or new_pos[1] > 7:
                        to_remove.append(move)
                for move in to_remove:
                    log[file][rank][i].remove(move)

        return log
    