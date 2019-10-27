from itertools import cycle

def display(C_B):
  print("{} | {} | {}".format(C_B[0],C_B[1],C_B[2]) )
  print("----------")
  print("{} | {} | {}".format(C_B[3],C_B[4],C_B[5]) )
  print("----------")
  print("{} | {} | {}".format(C_B[6],C_B[7],C_B[8]) )

def play_move(C_B,whose_turn,player_no):
  print("player " + player_no + " make your move....." )
  num = input("enter the number you wish to put your " + whose_turn)
  C_B[num] = whose_turn
  return C_B

def Check_Gameover(C_B):
	#current_board = arr[]
	#C_B = current_board
	# 1-> X wins
	#-1 -> O wins
	# 0 -> draw

    if( C_B[0] == C_B[4] == C_B[8] == "X"):
    	print("Player 1 won! , X marks the spot!")
    	return 1
    if( C_B[2] == C_B[4] == C_B[6] == "X"):
    	print("Player 1 won, X marks the spot!")
    	return 1
    if( C_B[0] == C_B[4] == C_B[8] == "O"):
    	print("Player 2 won! , O is the hero!")
    	return -1
    if( C_B[2] == C_B[4] == C_B[6] == "O"):
    	print("Player 2 won! , O is the hero!")
    	return -1

    for i in range(0,7,3):
      if(C_B[i] == C_B[i+1] ==C_B[i+2] == "X"):
        print("Player 1 won, X marks the spot!")
        return 1
    for i in range(3):
      if(C_B[i] == C_B[i+3] == C_B[i+2] == "O"):
        print("Player 2 won! , O is the hero!")
        return -1

    if( not " " not in C_B):
      print("seems you both are smart")
      return 0

    return False
	

def main_game():
  board = [ " " for i in range(9) ]
  display(board)

  whose_turn = cycle(["X","O"])
  player_no = cycle([1,2])

  while( not Check_Gameover(board)):
  	board = play_move(board,next(whose_turn),next(play_move))
  	display(board)



main_game()