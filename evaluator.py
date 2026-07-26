import torch
import numpy as np
from chess import Chess
from nneu import NNEU
from accumulator import Accumulator


class Evaluator():
    """
    The evaluator.
    """

    def __init__(self, file_path):
        self.device = (torch.device("mps") if torch.backends.mps.is_available() else 
                        torch.device("cuda") if torch.cuda.is_available() else 
                        torch.device("cpu")
        )

        assert file_path is not None, "Need the model weights."

        self.model = NNEU()
        self.model.to(self.device)
        saved_state_dict = torch.load(file_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(saved_state_dict)
        self.model.eval()

        print(f"loaded in model from ${file_path}")



    def get_best_move(self, game: Chess, maximizing: bool, depth_limit: int):
        """Get the best move to play."""
        moves = game.get_all_avaliable_moves()
        best_move = moves[0]

        if not moves:
            return None
        
        previous_score = 0
        d = 50
        
        for current_depth in range(1, depth_limit):
            ordered = [best_move] + [move for move in moves]
            
            if (current_depth == 1):
                alpha, beta = -float('inf'), float('inf')
            else:
                alpha, beta = previous_score - d, previous_score + d

            while True:
                current_best_move = ordered[0]
                current_best_move_score = -float("inf") if maximizing else float('inf')

                for start, end, promotion in ordered:
                    
                    if game.make_move(start, end, promotion):
                        score = self._minimax(game, -float("inf"), float('inf'), current_depth - 1, not maximizing)                    

                        if maximizing and score > current_best_move_score:
                            current_best_move, current_best_move_score = (start, end, promotion), score
                        elif not maximizing and score < current_best_move_score:
                            current_best_move, current_best_move_score = (start, end, promotion), score
                            
                        game.undo_move()

                if (current_best_move_score <= alpha):
                    alpha = previous_score - d * 2
                    d *= 2
                elif (current_best_move_score >= beta):
                    beta = previous_score + d *2
                    d *=2
                else:
                    break
                
                previous_score = current_best_move_score
                best_move = current_best_move
                d = 50
        
        return best_move
                

    def _evaluate(self, board):
        """
        Use the model to evaluate the board.
        """
        return -1
    
    def build_accumulator(self):
        """
        Set up the accumulator.
        """
        accumulator =  Accumulator(self.model.feature_extractor.weight.detach().cpu().numpy(), self.model.feature_bias.detach().cpu().numpy())
        return accumulator
    
    
    def _minimax(self, game : Chess, alpha: float, beta:float, depth, maximizing):
        """
        The adversarial search algorithm.
        """
        
        moves = game.get_all_avaliable_moves()

        if not moves:
            if game.is_in_check(game.board, game.turn, game.attack_directions, game.king_locations[0 if game.turn else 1]):
                return float("inf") if maximizing else -float("inf")
            return 0

        if depth == 0:
            if (game.is_in_check(game.board, game.turn, game.attack_directions, game.king_locations[0 if game.turn else 1])):
                return -float('inf') if maximizing else float("inf")
            return self._evaluate(game)
    
        if maximizing:
            for start, end, promotion in moves:
                if game.make_move(start, end ,promotion):
                    alpha = max(alpha, self._minimax(game, alpha, beta, depth - 1, False))
                    game.undo_move()
                    if beta <= alpha:
                        break
            return alpha
        
        else:
            for start, end, promotion in moves:
                if game.make_move(start, end, promotion):
                    beta = min(beta, self._minimax(game, alpha, beta, depth - 1, False))
                    game.undo_move()
                    if beta <= alpha:
                        break
            return beta
        