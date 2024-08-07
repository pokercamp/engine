'''
Encapsulates game and round state information for the player.
'''
from collections import namedtuple
import eval7
import itertools

from .actions import FoldAction, CheckAction, CallAction, RaiseAction

GameState = namedtuple('GameState', ['bankroll', 'game_clock', 'round_num'])
TerminalState = namedtuple('TerminalState', ['deltas', 'previous_state'])

NUM_ROUNDS = 1000
STARTING_STACK = 20
SMALL_BLIND = 1
BIG_BLIND = 2

def card_to_json(card):
    return {
        "rank": eval7.ranks[card.rank],
        "suit": eval7.suits[card.suit],
    }

def card_from_json(data):
    """Create a Card instance from JSON data."""
    card_string = data["rank"] + data["suit"]
    return eval7.Card(card_string)

class RoundState(namedtuple('_RoundState', ['turn_number', 'street', 'player_to_act', 'pips', 'stacks', 'hands', 'deck', 'n_ranks', 'n_streets', 'action_history', 'previous_state'])):
    def showdown(self):
        # don't compute this on the player side; at this point we don't even
        # have enough info to do so, but in the next few messages we'll get an
        # info message with the final hands, and after that a payoff message
        return self
    
    def visible_hands(self, seat, *, for_json=False):
        ret = [None for _ in self.hands]
        if for_json:
            ret[seat] = [card_to_json(card) for card in self.hands[seat]]
        else:
            ret[seat] = self.hands[seat]
        return ret
    
    def public(self):
        return None

    def legal_actions(self):
        active = self.player_to_act
        inactive = 1 - active
        
        if self.stacks[inactive] == 0:
            return {FoldAction, CallAction}
        
        raiseaction_if_legal = {RaiseAction} if self.raise_bounds()[1] > 0 else set()
        
        if self.pips[0] == self.pips[1]:
            match self.turn_number:
                case 0:
                    return {CheckAction} | raiseaction_if_legal
                case 1:
                    return {CallAction} | raiseaction_if_legal # a poker player would call this a check
                case _:
                    return set()
        
        assert self.pips[active] < self.pips[inactive]
        
        return {FoldAction, CallAction} | raiseaction_if_legal

    def raise_bounds(self):
        active = self.player_to_act
        inactive = 1 - active
        
        action_history_by_street = [
            list(y)
            for x, y
            in itertools.groupby(
                self.action_history,
                lambda a: isinstance(a, CallAction),
            )
            if not x
        ]
        
        if len(action_history_by_street) <= self.street:
            action_history_by_street.append([])
        
        min_raise = max(
            [BIG_BLIND]
            + [
                action.size
                for action
                in action_history_by_street[-1]
                if isinstance(action, RaiseAction)
            ]
        )
        max_raise = min(
            self.stacks[inactive],
            self.stacks[active] - (self.pips[inactive] - self.pips[active]),
        )
        
        return (
            min(min_raise, max_raise),
            max_raise,
        )

    def proceed_street(self):
        assert self.street <= self.n_streets
        
        if self.street == self.n_streets - 1:
            return self.showdown()
        
        return RoundState(
            turn_number=0,
            street=self.street+1,
            player_to_act=1, # every street after the preflop starts on the big blind
            pips=self.pips,
            stacks=self.stacks,
            hands=self.hands,
            deck=self.deck,
            n_ranks=self.n_ranks,
            n_streets=self.n_streets,
            action_history=self.action_history,
            previous_state=self,
        )

    def proceed(self, action):
        active = self.player_to_act
        inactive = 1 - active
        
        if isinstance(action, FoldAction):
            winner = inactive
            loser = active
        
            deltas = [0, 0]
            deltas[winner] = self.pips[loser]
            deltas[loser] = -self.pips[loser]
            
            return TerminalState(
                deltas,
                RoundState(
                    turn_number=self.turn_number + 1,
                    street=self.street,
                    player_to_act=None,
                    pips=self.pips,
                    stacks=self.stacks,
                    hands=self.hands,
                    deck=self.deck,
                    n_ranks=self.n_ranks,
                    n_streets=self.n_streets,
                    action_history=self.action_history + [action],
                    previous_state=self,
                ),
            )
        
        if isinstance(action, CheckAction):
            return RoundState(
                turn_number=self.turn_number + 1,
                street=self.street,
                player_to_act=1-self.player_to_act,
                pips=self.pips,
                stacks=self.stacks,
                hands=self.hands,
                deck=self.deck,
                n_ranks=self.n_ranks,
                n_streets=self.n_streets,
                action_history=self.action_history + [action],
                previous_state=self,
            )
        
        if isinstance(action, CallAction):
            assert self.pips[inactive] >= self.pips[active]
            amount = self.pips[inactive] - self.pips[active]
            assert amount <= self.stacks[active]
            
            pips = self.pips
            pips[active] += amount
            
            stacks = self.stacks
            stacks[active] -= amount
            
            if self.street == 0 and self.turn_number == 0:
                # Big Blind option
                return RoundState(
                    turn_number=self.turn_number + 1,
                    street=self.street,
                    player_to_act=1-self.player_to_act,
                    pips=pips,
                    stacks=stacks,
                    hands=self.hands,
                    deck=self.deck,
                    n_ranks=self.n_ranks,
                    n_streets=self.n_streets,
                    action_history=self.action_history + [action],
                    previous_state=self,
                )
            else:
                # go to next street
                return RoundState(
                    turn_number=self.turn_number + 1,
                    street=self.street,
                    player_to_act=None,
                    pips=pips,
                    stacks=stacks,
                    hands=self.hands,
                    deck=self.deck,
                    n_ranks=self.n_ranks,
                    n_streets=self.n_streets,
                    action_history=self.action_history + [action],
                    previous_state=self,
                ).proceed_street()
        
        if isinstance(action, RaiseAction):
            assert self.pips[inactive] >= self.pips[active]
            (min_raise, max_raise) = self.raise_bounds()
            assert min_raise <= action.size and action.size <= max_raise
            amount = self.pips[inactive] - self.pips[active] + action.size
            assert amount <= self.stacks[active]
            
            pips = self.pips
            pips[active] += amount
            
            stacks = self.stacks
            stacks[active] -= amount
            
            return RoundState(
                turn_number=self.turn_number + 1,
                street=self.street,
                player_to_act=1-self.player_to_act,
                pips=pips,
                stacks=stacks,
                hands=self.hands,
                deck=self.deck,
                n_ranks=self.n_ranks,
                n_streets=self.n_streets,
                action_history=self.action_history + [action],
                previous_state=self,
            )
