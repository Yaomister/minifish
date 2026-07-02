import torch
import numpy as np
from chess import Chess
from nneu import NNEU
from accumulator import Accumulator


class Evaluator():

    def __init__(self, file_path = None):
        model = NNEU()
        self.device = (torch.device("mps") if torch.backends.mps.is_available() else 
                        torch.device("cuda") if torch.cuda.is_available() else 
                        torch.device("cpu")
        )

        if (file_path):
            model.to(self.device)
            saved_state_dict = torch.load(file_path)
            model.load_state_dict(saved_state_dict)
            model.eval()
            print(f"loaded in model from ${file_path}")

        self.model = model


    def get_best_move(self, game: Chess, maximizing: bool, depth_limit: int):


        moves = game.get_all_avaliable_moves()
        best_move = moves[0]

        if not moves:
            return None
        
        previous_score = 0
        d = 50
        
        for current_depth in range(1, depth_limit):
            ordered = [best_move] + [move for move in moves]
            
            if (current_depth == 1):
                alpha, beta = float('inf'), -float('inf')
            else:
                alpha, beta = previous_score - d, previous_score + d

            while True:
                current_best_move = ordered[0]
                current_best_move_score = -float("inf") if maximizing else float('inf')

                for start, end, promotion in ordered:
                    game.make_move(start, end, promotion)
                    score = self._minimax(game, float("inf"), -float('inf'), current_depth - 1, not maximizing)
                    game.undo_move()

                    if maximizing and score > current_best_move_score:
                        current_best_move, current_best_move_score = (start, end, promotion), score
                    elif not maximizing and score < current_best_move_score:
                        current_best_move, current_best_move_score = (start, end, promotion), score

                if (current_best_move_score <= alpha):
                    alpha = previous_score - d * 2
                    d *= 2
                elif (current_best_move >= beta):
                    beta = previous_score + d *2
                    d *=2
                else:
                    break
                
                previous_score = current_best_move_score
                best_move = current_best_move
                d = 50
        
        return best_move
                


    def _evaluate(self, board):
    
        return -1
    
    
    def _minimax(self, game : Chess, alpha: float, beta:float, depth, maximizing):

        moves = game.get_all_avaliable_moves()

        if not moves:
            if game.is_in_check():
                return float("inf") if maximizing else -float("inf")
            return 0

        if depth == 0:
            if (game.is_in_check(game.board, game.white_moves, game.attack_directions, game.king_locations[0 if game.white_moves else 1])):
                return -float('inf') if maximizing else float("inf")
            return self._evaluate(game)
    
        if maximizing:
            for start, end, promotion in moves:
                game.make_move(start, end ,promotion)
                alpha = max(alpha, self._minimax(game, alpha, beta, depth - 1, False))
                self.model.undo_move()
                if beta <= alpha:
                    break
            return alpha
        
        else:
            for start, end, promotion in moves:
                game.make_move(start, end, promotion)
                beta = min(beta, self._minimax(game, alpha, beta, depth - 1, False))
                self.model.undo_moves()
                if beta <= alpha:
                    break
            return beta
        
    def build_accumulator(self):
        accumulator =  Accumulator(self.model.feature_extractor.weight.detach().cpu().numpy(), self.model.feature_extractor.bias.detach().cpu().numpy())
        return accumulator