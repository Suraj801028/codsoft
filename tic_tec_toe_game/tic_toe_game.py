import sys
import numpy as np
import pygame
pygame.init()
pygame.font.init()
#colors
white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 0)
grey = (128, 128, 128)
black = (0, 0, 0)
#properties
widht = 400
height = 400
line_width = 10
board_rows = 3
board_cols = 3
square_size = widht // board_cols
circle_radius = square_size // 3
circle_width = 15
cross_width = 25
 
#screen
screen = pygame.display.set_mode((widht, height))
pygame.display.set_caption('Tic Tac Toe')
screen.fill(white)

game_message = ""
# font for messages
font = pygame.font.SysFont(None, 36)
#board
board = np.zeros((board_rows, board_cols))
def draw_lines(color=grey):
    for i in range(1, board_rows):
        pygame.draw.line(screen, color, (0, square_size * i), (widht, square_size * i), line_width) 
        pygame.draw.line(screen, color, (square_size * i, 0), (square_size * i, height), line_width)

def draw_figures():
    for row in range(board_rows):
        for col in range(board_cols):
            if board[row][col] == 1:
                pygame.draw.circle(screen, red, (int(col * square_size + square_size // 2), int(row * square_size + square_size // 2)), circle_radius, circle_width)
            elif board[row][col] == 2:
                pygame.draw.line(screen, green, (col * square_size + cross_width, row * square_size + cross_width), (col * square_size + square_size - cross_width, row * square_size + square_size - cross_width), cross_width) 
                pygame.draw.line(screen, green, (col * square_size + cross_width, row * square_size + square_size - cross_width), (col * square_size + square_size - cross_width, row * square_size + cross_width), cross_width)        
   
def mark_square(row, col, player):
    board[row][col] = player

def available_square(row, col):
    return board[row][col] == 0
def is_board_full(check_board=board):
    for row in range(board_rows):
        for col in range(board_cols):
            if check_board[row][col] == 0:
                return False
    return True

def check_win(player, check_board=board):
    for col in range(board_cols):
        if check_board[0][col] == player and check_board[1][col] == player and check_board[2][col] == player:
        
            return True
    for row in range(board_rows):
        if check_board[row][0] == player and check_board[row][1] == player and check_board[row][2] == player:
        
            return True
    if check_board[2][0] == player and check_board[1][1] == player and check_board[0][2] == player:
    
        return True
    if check_board[0][0] == player and check_board[1][1] == player and check_board[2][2] == player:

        return True
    return False

def minimax(board, depth, is_maximizing):
    if check_win(2, board):
        return float('inf')
    elif check_win(1, board):
        return float('-inf')
    elif is_board_full(board):
        return 0

    if is_maximizing:
        best_score = -sys.maxsize
        for row in range(board_rows):
            for col in range(board_cols):
                if board[row][col] == 0:
                    board[row][col] = 2
                    score = minimax(board, depth + 1, False)
                    board[row][col] = 0
                    best_score = max(score, best_score)
        return best_score
    else:
        best_score = sys.maxsize
        for row in range(board_rows):
            for col in range(board_cols):
                if board[row][col] == 0:
                    board[row][col] = 1
                    score = minimax(board, depth + 1, True)
                    board[row][col] = 0
                    best_score = min(score, best_score)
        return best_score
def ai_move():
    best_score = -sys.maxsize
    best_move = None
    for row in range(board_rows):
        for col in range(board_cols):
            if board[row][col] == 0:
                board[row][col] = 2
                score = minimax(board, 0, False)
                board[row][col] = 0
                if score > best_score:
                    best_score = score
                    best_move = (row, col)
    if best_move is not None:
        mark_square(best_move[0], best_move[1], 2)

def restart_game():
    screen.fill(white)
    draw_lines()
    for row in range(board_rows):
        for col in range(board_cols):
            board[row][col] = 0
    global game_message
    game_message = ""


def draw_message(text, color=black):
    if not text:
        return
    rendered = font.render(text, True, color)
    rect = rendered.get_rect(center=(widht // 2, height // 2))
    screen.blit(rendered, rect)

draw_lines()
player = 1
game_over = False

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
            mouseX = event.pos[0]
            mouseY = event.pos[1]
            clicked_row = int(mouseY // square_size)
            clicked_col = int(mouseX // square_size)
            if available_square(clicked_row, clicked_col):
                mark_square(clicked_row, clicked_col, player)
                if check_win(player):
                    game_over = True
                    game_message = f"Player {int(player)} wins!"
                if not game_over:
                    ai_move()
                    if check_win(2):
                        game_over = True
                        game_message = "Player 2 wins!"
                if not game_over:
                    if is_board_full():
                        game_over = True
                        game_message = "Draw!"

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                restart_game()
                game_over = False
                player = 1

    if not game_over:
        draw_figures()
    else:
        if check_win(1):
            draw_figures()
            draw_lines(green)
        elif check_win(2):
            draw_figures()
            draw_lines(red)
        else:
            draw_figures()
            draw_lines(grey)

    # draw end-of-game message (if any)
    draw_message(game_message)

    pygame.display.update()