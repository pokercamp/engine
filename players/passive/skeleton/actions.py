'''
The actions that the player is allowed to take.
'''
from collections import namedtuple

class FoldAction(namedtuple('FoldAction', [])):
    verb = 'Fold'
    def __hash__(self):
        return hash('FoldAction')
    def __eq__(self, other):
        return isinstance(other, FoldAction)
    def __repr__(self):
        return 'Fold'

class CheckAction(namedtuple('CheckAction', [])):
    verb = 'Check'
    def __hash__(self):
        return hash('CheckAction')
    def __eq__(self, other):
        return isinstance(other, CheckAction)
    def __repr__(self):
        return 'Check'

# also use this for the last player of a round checking
class CallAction(namedtuple('CallAction', [])):
    verb = 'Call'
    def __hash__(self):
        return hash('CallAction')
    def __eq__(self, other):
        return isinstance(other, CallAction)
    def __repr__(self):
        return 'Call'

# also use this for Bet
class RaiseAction(namedtuple('RaiseAction', ['size'])):
    verb = 'Raise'
    def __hash__(self):
        return hash(f'RaiseAction({self.size})')
    def __eq__(self, other):
        return isinstance(other, RaiseAction)
    def __repr__(self):
        return f'Raise({self.size})'
