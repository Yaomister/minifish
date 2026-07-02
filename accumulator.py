import numpy as np


class Accumulator():
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases
        self.accumulator = {True: biases.copy(), False: biases.copy()}
        self.square_with_king = {True:None, False:None}

    @staticmethod
    def get_index(square_with_king, square_with_piece, piece_type, is_white, perspective):
        piece_weights = {
            "P": 0,
            "N": 1,
            "B": 2,
            "R": 3,
            "Q": 4
        }
        if not perspective:
            square_with_king = square_with_king ^ 56
            square_with_piece = square_with_piece ^ 56

        piece = piece_weights[piece_type] * 2 + (0 if is_white else 1)

        return square_with_king * 640 + square_with_piece * 10 + piece
    
    def _refresh_accumulation(self, board, perspective):
        for (file, rank), square in np.ndenumerate(board):
            if square and square[0] == "K" and square[1] == perspective:
                self.square_with_king[perspective] = file * 8 + rank
                break
        
        accumulator = self.biases.copy()
        for (file, rank), square in np.ndenumerate(board):
            if square and square[0] != "K":
                index = self.get_index(self.square_with_king[perspective], file * 8 + rank, square[0], square[1], perspective)
                accumulator[perspective] += self.weights[:, index]

        self.accumulator[perspective] = accumulator

    
    def _convert_coordinates_to_tensor_index(self, square):
        return square[0] * 8 + square[1]
    
    def refresh(self, board):
        self._refresh_accumulation(board, True)
        self._refresh_accumulation(board, False)
        return
    
    def apply_difference(self, added, removed):
        for perspective in (True, False):
            square_with_king = self.square_with_king[perspective]
            for square, piece, is_white in added:
                self.accumulator[perspective] += self.weights[:, self.get_index(square_with_king, self._convert_coordinates_to_tensor_index(square), piece, is_white, perspective)]
            for square, piece, is_white in removed:
                self.accumulator[perspective] -= self.weights[:, self.get_index(square_with_king, self._convert_coordinates_to_tensor_index(square), piece, is_white, perspective)]

    def get_logits(self, perspective):
        return np.concatenate(self.accumulator[perspective], self.accumulator[not perspective])