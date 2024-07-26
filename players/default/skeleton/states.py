'''
Encapsulates game and round state information for the player.
'''
from collections import namedtuple
from .actions import FoldAction, CallAction, CheckAction, RaiseAction

GameState = namedtuple('GameState', ['bankroll', 'game_clock', 'round_num'])
TerminalState = namedtuple('TerminalState', ['deltas', 'previous_state'])

NUM_ROUNDS = 1000
STARTING_STACK = 50
ANTE = 1
BET_SIZE = 2

class RoundState(namedtuple('_RoundState', ['turn_number', 'street', 'pips', 'stacks', 'hands', 'deck', 'action_history', 'previous_state'])):
    # @staticmethod
    # def new():
    #     '''
    #     Returns a RoundState representing the start of a Leduc game.
    #     '''
    #     deck = LeducDeck()
    #     hands = [deck.deal(), deck.deal()]
    #     pips = [ANTE, ANTE]
    #     stacks = [STARTING_STACK - ANTE, STARTING_STACK - ANTE]
    #     return RoundState(
    #         turn_number = 0,
    #         street = 0,
    #         pips = pips,
    #         stacks = stacks,
    #         hands = hands,
    #         deck = deck,
    #         action_history = [],
    #         previous_state = None,
    #     )
    
    def showdown(self):
        # don't compute this on the player side; at this point we don't even
        # have enough info to do so, but in the next few messages we'll get an
        # info message with the final hands, and after that a payoff message
        return self
    
    # def visible_hands(self, seat):
    #     ret = [None for _ in self.hands]
    #     ret[seat] = self.hands[seat]
    #     return ret
    
    def public(self):
        return {}

    def legal_actions(self):
        '''
        Returns a set which corresponds to the active player's legal moves.
        '''
        active = self.turn_number % 2
        call_cost = self.pips[1-active] - self.pips[active]
        # print('cont cost: ', call_cost)
        if call_cost == 0:
            if self.stacks[0] == 0: # P0 is all-in
                # consider whether this should be unreachable / return empty set
                return {CheckAction}
            elif self.stacks[1] == 0: # P1 is all-in
                # consider whether this should be unreachable / return empty set
                return {CheckAction}
            else:
                return {CheckAction, RaiseAction}
        else: # call_cost > 0
            if self.stacks[1-active] == 0: # opponent went all-in, so we can't re-raise
                return {FoldAction, CallAction}
            if call_cost >= self.stacks[active]: # calling would put us all-in, so we can't re-raise
                return {FoldAction, CallAction}
            else:
                # RaiseAction includes an all-in shove less than the default/minimum bet
                return {FoldAction, CallAction, RaiseAction}

    def raise_bounds(self):
        '''
        Returns the legal raise size.
        '''
        active = self.turn_number % 2
        if self.street == 0:
            default_bet_amount = BET_SIZE
        else: #self.street == 1
            default_bet_amount = BET_SIZE*2
        call_cost = self.pips[1-active] - self.pips[active]

        assert call_cost <= self.stacks[active] # should have been enforced by previous state's legal_actions
        if call_cost == self.stacks[active]:
            return 0

        max_contribution = min(
            default_bet_amount, # default bet
            self.stacks[1-active], # put our opponent all-in
            self.stacks[active] - call_cost, # go all-in; positive because of above assertion
        )
        # min_contribution = min(default_bet_amount, max_contribution)
        return max_contribution

    def proceed_street(self):
        '''
        Resets the players' pips and advances the game tree to the next round of betting.

        Current logic: there's a final RoundState at the end of the prev_street, and then
        a different RoundState to begin the next_street. This is weird because the
        end-of-prev_street RoundState has no actions to take. You should never encounter
        that state by calling state.proceed() because proceed() will skip over it. You'll
        only ever see it if you look in state.previous_state.
        '''
        assert self.pips[0] == self.pips[1]
        
        if self.street == 0:
            return RoundState(
                turn_number = 0,
                street = 1,
                pips = self.pips,
                stacks = self.stacks,
                hands = self.hands,
                deck = self.deck,
                action_history = self.action_history,
                previous_state = self,
            )
        else:
            return self.showdown()

    def proceed(self, action):
        active = self.turn_number % 2
        inactive = 1 - active
        new_pips = list(self.pips)
        new_stacks = list(self.stacks)
        
        if isinstance(action, FoldAction):
            deltas = [0, 0]
            winner = inactive
            loser = active
            deltas[winner] = self.pips[loser]
            deltas[loser] = -self.pips[loser]

            return TerminalState(
                deltas,
                RoundState(
                    turn_number = self.turn_number + 1,
                    street = self.street,
                    pips = self.pips,
                    stacks = self.stacks,
                    hands = self.hands,
                    deck = self.deck,
                    action_history = self.action_history + [FoldAction()],
                    previous_state = self,
                )
            )
        
        elif isinstance(action, CallAction):
            call_cost = new_pips[inactive] - new_pips[active]
            new_stacks[active] -= call_cost
            new_pips[active] += call_cost
            state = RoundState(
                turn_number = self.turn_number + 1,
                street = self.street,
                pips = new_pips,
                stacks = new_stacks,
                hands = self.hands,
                deck = self.deck,
                action_history = self.action_history + [CallAction()],
                previous_state = self,
            )
            return state.proceed_street()
        
        elif isinstance(action, CheckAction):
            state = RoundState(
                turn_number = self.turn_number + 1,
                street = self.street,
                pips = self.pips,
                stacks = self.stacks,
                hands = self.hands,
                deck = self.deck,
                action_history = self.action_history + [CheckAction()] if isinstance(self.action_history, list) else [CheckAction()],
                previous_state = self,
            )

            if self.turn_number == 0:
                return state
            elif self.turn_number == 1:  # check-check
                assert isinstance(self.action_history[-1], CheckAction)
                return state.proceed_street()
            
            raise ValueError(f'proceeded with CheckAction on turn {self.turn_number}, expected 0 or 1')
        
        elif isinstance(action, RaiseAction):
            # only one size of raise in this game; self.raise_bounds() calculates it
            contribution = self.raise_bounds() + (self.pips[inactive] - self.pips[active])
            new_stacks[active] -= contribution
            new_pips[active] += contribution
            return RoundState(
                turn_number = self.turn_number + 1,
                street = self.street,
                pips = new_pips,
                stacks = new_stacks,
                hands = self.hands,
                deck = self.deck,
                action_history = self.action_history + [RaiseAction()],
                previous_state = self,
            )
        
        else:
            raise NotImplementedError(f'action ({action}) of unknown type')


# class RoundState(namedtuple('_RoundState', ['turn_number', 'street', 'pips', 'stacks', 'hands', 'deck', 'action_history', 'previous_state'])):
#     def showdown(self):
#         hands = self.hands
#         assert all(h is not None for h in self.hands)
#         street = self.street
#         community = self.deck.peek(street)
#         pips = self.pips

#         if hands[0] == community:
#             winner = 0
#         elif hands[1] == community:
#             winner = 1
#         elif hands[0] > hands[1]:
#             winner = 0
#         elif hands[1] > hands[0]:
#             winner = 1
#         else:
#             winner = -1 #Tie 

#         deltas = [0, 0]
#         if winner == 0 or winner == 1:
#             loser = 1 - winner
#             deltas[winner] = pips[loser]
#             deltas[loser] = -pips[loser]
        
#         return TerminalState(deltas, self)  
    
#     def legal_actions(self):
#         active = self.turn_number % 2
#         continue_cost = self.pips[1-active] - self.pips[active]
#         # print('cont cost: ', continue_cost)
#         if continue_cost == 0:
#             # we can only raise the stakes if both players can afford it
#             bets_forbidden = (self.stacks[0] == 0 or self.stacks[1] == 0)
#             return {CheckAction} if bets_forbidden else {CheckAction, RaiseAction}
#         # continue_cost > 0
#         # similarly, re-raising is only allowed if both players can afford it
#         raises_forbidden = (continue_cost >= self.stacks[active] or self.stacks[1-active] == 0)
#         print('raises forbidden: ', raises_forbidden)
#         # a = {FoldAction, CallAction} if raises_forbidden else {FoldAction, CallAction, RaiseAction}
#         # print('a': a)
#         # return a
#         return {FoldAction, CallAction} if raises_forbidden else {FoldAction, CallAction, RaiseAction}

#     def raise_bounds(self):
#         '''
#         Returns the legal raise size
#         '''
#         active = self.turn_number % 2
#         if self.street == 0:
#             bet_amount = BET_SIZE
#         else: #self.street == 1
#             bet_amount = BET_SIZE*2
#         continue_cost = self.pips[1-active] - self.pips[active] #bet_amount or 0 
#         max_contribution = min(bet_amount, self.stacks[1-active] + continue_cost, self.stacks[active])
#         #valid in case they went allin with small amount
#         # min_contribution = min(max_contribution, continue_cost + bet_amount)
#         return max_contribution #+self.pips[active]
                
#     def proceed_street(self):
#         '''
#         Resets the players' pips and advances the game tree to the next round of betting.
#         '''
#         if self.street == 0:
#             return RoundState(0, 1, [0, 0], self.stacks, self.hands, self.deck, self)
#         return self.showdown()

#     def proceed(self, action):
#         active = self.turn_number % 2
#         inactive = 1 - active
#         new_pips = list(self.pips)
#         new_stacks = list(self.stacks)
        
#         if isinstance(action, FoldAction):
#             delta = self.stacks[0] - STARTING_STACK if active == 0 else STARTING_STACK - self.stacks[1]
#             return TerminalState([delta, -delta], self)
        
#         if isinstance(action, CallAction):
#             contribution = new_pips[inactive] - new_pips[active]
#             new_stacks[active] -= contribution
#             new_pips[active] += contribution
#             state = RoundState(self.turn_number + 1, self.street, self.final_street, new_pips, new_stacks, self.hands, self.deck, self)
#             return state.proceed_street()
        
#         if isinstance(action, CheckAction):
#             if self.turn_number == 1:  # both players acted
#                 return self.proceed_street()
#             # let opponent act
#             return RoundState(self.turn_number + 1, self.street, self.pips, self.stacks, self.hands, self.deck, self)
#         # isinstance(action, RaiseAction)
#         contribution = action.amount - new_pips[active]
#         new_stacks[active] -= contribution
#         new_pips[active] += contribution
#         return RoundState(self.turn_number + 1, self.street, new_pips, new_stacks, self.hands, self.deck, self)