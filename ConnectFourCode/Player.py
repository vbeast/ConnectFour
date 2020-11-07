import math
import random

import numpy as np

class AIPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)

    def get_alpha_beta_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the alpha-beta pruning algorithm

        This will play against either itself or a human player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        column_pick, minimax_score = self.minimax(board, 4, -100000000000001, 100000000000001, True)
        return column_pick


    def minimax(self, board, depth, player_score, opp_score, maximizing_player):
        opponent_number = 0
        if self.player_number == 1:
            opponent_number = 2
        if self.player_number == 2:
            opponent_number = 1

        valid_columns = self.get_valid_locations(board)

        if depth == 0 or len(valid_columns) == 0:
            if depth == 0:
                return None, self.evaluation_function(board)
            if len(valid_columns) == 0:
                return None, 0

        if maximizing_player:
            value = -math.inf
            column = 0
            for col in valid_columns:
                board_copy = np.copy(board)
                new_board = self.drop_chip(board_copy, col, self.player_number)
                new_score = self.minimax(new_board, depth - 1, player_score, opp_score, False)
                if new_score[1] > value:
                    value = new_score[1]
                    column = col
                if player_score < value:
                    player_score = value
                if player_score >= opp_score:
                    break
            return column, value
        else:
            value = math.inf
            column = 0
            for col in valid_columns:
                board_copy = np.copy(board)
                new_board = self.drop_chip(board_copy, col, opponent_number)
                new_score = self.minimax(new_board, depth - 1, player_score, opp_score, True)
                if new_score[1] < value:
                    value = new_score[1]
                    column = col
                if opp_score > value:
                    opp_score = value
                if player_score >= opp_score:
                    break
            return column, value

    def get_expectimax_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the expectimax algorithm.

        This will play against the random player, who chooses any valid move
        with equal probability

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        column_pick, expectiminimax_score = self.expectiminimax(board, 4, True)
        return column_pick


    def expectiminimax(self, board, depth, maximizing_player):
        opponent_number = 0
        if self.player_number == 1:
            opponent_number = 2
        if self.player_number == 2:
            opponent_number = 1

        valid_columns = self.get_valid_locations(board)

        if depth == 0 or len(valid_columns) == 0:
            if depth == 0:
                return None, self.evaluation_function(board)
            if len(valid_columns) == 0:
                return None, 0

        if maximizing_player:
            value = -math.inf
            column = 0
            for col in valid_columns:
                board_copy = np.copy(board)
                new_board = self.drop_chip(board_copy, col, self.player_number)
                new_score = self.expectiminimax(new_board, depth - 1, False)
                if new_score[1] > value:
                    value = new_score[1]
                    column = col
            return column, value
        else:
            value = 0
            column = 0
            for col in valid_columns:
                board_copy = np.copy(board)
                new_board = self.drop_chip(board_copy, col, opponent_number)
                new_score = self.expectiminimax(new_board, depth - 1, True)
                value += new_score[1]/7

            return None, value

    def winning_board(self, board, piece):
        for row in board:
            row_list = list(row)
            if self.find_consecutive_chips(row_list, piece) == 3:
                return True

        column_list = board.T

        for col in column_list:
            column = list(col)
            if self.find_consecutive_chips(column, piece) == 3:
                return True

        return False

    def find_consecutive_chips(self, window, player_piece):
        # [0,2,2,2]
        count = 0
        condition_met = False
        for i in range(1, len(window)):
            if window[i-1] == player_piece and window[i] == player_piece:
                condition_met = True
                count += 1
        if condition_met:
            count += 1
        else:
            count = 0

        return count

    def score_window(self, window, player_piece, opponent_number):
        running_score = 0
        if window.count(opponent_number) > 0:
            if self.find_consecutive_chips(window, opponent_number) == 3 and window.count(0) == 1:
                running_score -= 800
            else:
                return 0
        else:
            if window.count(player_piece) == 1:
                running_score += 1
            if window.count(player_piece) == 2 or self.find_consecutive_chips(window, player_piece) == 2:
                if window.count(player_piece) == 2:
                    running_score += 30
                if self.find_consecutive_chips(window, player_piece) == 2:
                    running_score += 50
            if window.count(player_piece) == 3 or self.find_consecutive_chips(window, player_piece) == 2:
                if window.count(player_piece) == 3:
                    running_score += 100
                if self.find_consecutive_chips(window, player_piece) == 3:
                    running_score += 200
            if window.count(player_piece) == 4:
                running_score += 100000

        return running_score



    def get_utility_horizontal(self, board, player_piece, opponent_number):
        running_score = 0
        for col in board:
            current_col = col

            for i in range(0, len(current_col)-3):
                window = list(current_col[i:i+4])

                running_score += self.score_window(window, player_piece, opponent_number)

        return running_score

    def get_utility_vertical(self, board, player_piece, opponent_number):
        running_score = 0
        for col in board:
            current_col = col
            for i in range(0, len(current_col) - 3):
                window = list(current_col[i:i + 4])

                running_score += self.score_window(window, player_piece, opponent_number)

        return running_score

    def get_utility_diagonal(self, board, player_piece, opponent_number):
        running_score = 0
        for i in range(len(board) - 3):
            for j in range(len(board[0]) - 3):
                window = []
                for k in range(0, 4):
                    window.append(board[i + k][j + k])
                    running_score += self.score_window(window, player_piece, opponent_number)

        for i in range(len(board) - 3):
            for j in range(len(board[0]) - 3):
                window = []
                for k in range(0, 4):
                    window.append(board[i + 3 - k][j + k])
                    running_score += self.score_window(window, player_piece, opponent_number)

        return running_score

    def evaluation_function(self, board):
        """
        Given the current stat of the board, return the scalar value that 
        represents the evaluation function for the current player
       
        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The utility value for the current board
        """
        opponent_number = 0
        if self.player_number == 1:
            opponent_number = 2
        if self.player_number == 2:
            opponent_number = 1

        utility = 0

        utility += self.get_utility_horizontal(board, self.player_number, opponent_number)

        column_list = np.copy(board.T)

        utility = utility + self.get_utility_vertical(column_list, self.player_number, opponent_number)

        utility += self.get_utility_diagonal(board, self.player_number, opponent_number)

        board_utility = utility

        return board_utility

    # return a list of all valid locations in a board
    # ie. the columns that still have zeros in them
    def get_valid_locations(self, board):
        valid_columns = []
        column_list = board.T
        for i in range(0, len(column_list)):
            col = list(column_list[i])
            if col.count(0) > 0:
                valid_columns.append(i)
        return valid_columns

    # simulates dropping a chip into a slot of a
    # board copy and returns the new matrix
    def drop_chip(self, board, column, chip):
        board_copy = np.copy(board)
        column_list = np.copy(board_copy.T)

        selected_column = column_list[column]

        for i in range(len(selected_column)-1, 0, -1):
            if selected_column[i] == 0:
                selected_column[i] = chip
                column_list[column] = selected_column
                break
        board_copy = column_list.T

        return board_copy


class RandomPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'random'
        self.player_string = 'Player {}:random'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state select a random column from the available
        valid moves.

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:,col]:
                valid_cols.append(col)

        return np.random.choice(valid_cols)


class HumanPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'human'
        self.player_string = 'Player {}:human'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state returns the human input for next move

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)

        move = int(input('Enter your move: '))

        while move not in valid_cols:
            print('Column full, choose from:{}'.format(valid_cols))
            move = int(input('Enter your move: '))

        return move

