'''
Encapsulates game and round state information for the player.
'''
from collections import namedtuple
from .actions import FoldAction, CallAction, CheckAction, RaiseAction

GameState = namedtuple('GameState', ['bankroll', 'game_clock', 'round_num'])
TerminalState = namedtuple('TerminalState', ['deltas', 'previous_state'])

NUM_ROUNDS = 1000
STARTING_STACK = 400
ANTE = 1
BET_SIZE = 2

class RoundState(namedtuple('_RoundState', ['turn_number', 'street', 'pips', 'stacks', 'hands', 'deck', 'action_history', 'previous_state'])):
    def showdown(self):
        hands = self.hands
        street = self.street
        community = self.deck.peek(street)
        pips = self.pips

        if hands[0] == community:
            winner = 0
        elif hands[1] == community:
            winner = 1
        elif hands[0] > hands[1]:
            winner = 0
        elif hands[1] > hands[0]:
            winner = 1
        else:
            winner = -1 #Tie 

        deltas = [0, 0]
        if winner == 0 or winner == 1:
            loser = 1 - winner
            deltas[winner] = pips[loser]
            deltas[loser] = -pips[loser]
        
        return TerminalState(deltas, self)  
    
    def legal_actions(self):
        active = self.turn_number % 2
        continue_cost = self.pips[1-active] - self.pips[active]
        # print('cont cost: ', continue_cost)
        if continue_cost == 0:
            # we can only raise the stakes if both players can afford it
            bets_forbidden = (self.stacks[0] == 0 or self.stacks[1] == 0)
            return {CheckAction} if bets_forbidden else {CheckAction, RaiseAction}
        # continue_cost > 0
        # similarly, re-raising is only allowed if both players can afford it
        raises_forbidden = (continue_cost >= self.stacks[active] or self.stacks[1-active] == 0)
        print('raises forbidden: ', raises_forbidden)
        # a = {FoldAction, CallAction} if raises_forbidden else {FoldAction, CallAction, RaiseAction}
        # print('a': a)
        # return a
        return {FoldAction, CallAction} if raises_forbidden else {FoldAction, CallAction, RaiseAction}

    def raise_bounds(self):
        '''
        Returns the legal raise size
        '''
        active = self.turn_number % 2
        if self.street == 0:
            bet_amount = BET_SIZE
        else: #self.street == 1
            bet_amount = BET_SIZE*2
        continue_cost = self.pips[1-active] - self.pips[active] #bet_amount or 0 
        max_contribution = min(bet_amount, self.stacks[1-active] + continue_cost, self.stacks[active])
        #valid in case they went allin with small amount
        # min_contribution = min(max_contribution, continue_cost + bet_amount)
        return max_contribution #+self.pips[active]
                
    def proceed_street(self):
        '''
        Resets the players' pips and advances the game tree to the next round of betting.
        '''
        if self.street == 0:
            return RoundState(0, 1, [0, 0], self.stacks, self.hands, self.deck, self)
        return self.showdown()

    def proceed(self, action):
        active = self.turn_number % 2
        inactive = 1 - active
        new_pips = list(self.pips)
        new_stacks = list(self.stacks)
        
        if isinstance(action, FoldAction):
            delta = self.stacks[0] - STARTING_STACK if active == 0 else STARTING_STACK - self.stacks[1]
            return TerminalState([delta, -delta], self)
        
        if isinstance(action, CallAction):
            contribution = new_pips[inactive] - new_pips[active]
            new_stacks[active] -= contribution
            new_pips[active] += contribution
            state = RoundState(self.turn_number + 1, self.street, self.final_street, new_pips, new_stacks, self.hands, self.deck, self)
            return state.proceed_street()
        
        if isinstance(action, CheckAction):
            if self.turn_number == 1:  # both players acted
                return self.proceed_street()
            # let opponent act
            return RoundState(self.turn_number + 1, self.street, self.pips, self.stacks, self.hands, self.deck, self)
        # isinstance(action, RaiseAction)
        contribution = action.amount - new_pips[active]
        new_stacks[active] -= contribution
        new_pips[active] += contribution
        return RoundState(self.turn_number + 1, self.street, new_pips, new_stacks, self.hands, self.deck, self)