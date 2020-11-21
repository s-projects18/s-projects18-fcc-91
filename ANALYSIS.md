## RULES ##
- P > R | R > S | S > P | P=P,R=R,S=S
- every move has 1 win and 1 loose and 1 tie (eg R=R)
- player does not know his enemy (only indirect by strategy)

## TARGET ##
- To pass this challenge your program must play matches against four different bots, winning at least 60% of the games in each match

## BOT-STRATEGIES ##
- LEVEL 1: abbey: always reacts on the last move
 me: P -> abbey: S etc
- kris: similar abbey
- LEVEL 2: quincy: random move, no dependency
- LEVEL 3: mrugesh: always P

## ALGORYTHMICAL SOLUTION ##
- default: random
- try to identify a type while playing

## NEURONAL SOLUTION ##
### output layer ###
- a move (= prediction based on input layer)
- 3 possible moves (P,R,S) -> 3 nodes
- node with highest probability = best move

### input layer ###
- also 3 nodes with PRS-weights: 1 0 0, 0 1 0, 0 0 1
- I get the last opponents move
- whatever I played before is not important as I can judge the value of each opponent-move
- idea: lerning while playing. new learning for each player
- not: one network for all player

### hidden layer ###
- ?

### model ###
- ?

