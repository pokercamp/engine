'''
The infrastructure for interacting with the engine.
'''
import argparse
import json
import socket
from .actions import FoldAction, CheckAction, CallAction, RaiseAction
from .states import GameState, TerminalState, RoundState
from .states import STARTING_STACK, SMALL_BLIND, BIG_BLIND
from .bot import Bot

DECODE = {'Fold': FoldAction, 'Check': CheckAction, 'Call': CallAction, 'Raise': RaiseAction}

class Runner():
    '''
    Interacts with the engine.
    '''

    def __init__(self, pokerbot, socketfile):
        self.pokerbot = pokerbot
        self.socketfile = socketfile

    def receive(self):
        '''
        Generator for incoming messages from the engine.
        '''
        while True:
            packet = self.socketfile.readline().strip()
            if not packet:
                break
            yield packet

    def send(self, action, seat):
        '''
        Encodes an action and sends it to the engine.
        '''
        self.socketfile.write(json.dumps({
            'type': 'action',
            'action': {
                'verb': action.verb,
                **({'size': action.size} if isinstance(action, RaiseAction) else {})
            },
            'player': seat,
        }) + '\n')
        self.socketfile.flush()

    def run(self):
        '''
        Reconstructs the game tree based on the action history received from the engine.
        '''
        game_state = GameState(0, 0., 1)
        round_state = None
        seat = 0
        for packet in self.receive():
            # okay to accept a single json object
            if packet[0] == '{':
                packet = f'[{packet}]'
            for message in json.loads(packet):
                try:
                    match message['type']:
                        case 'hello':
                            pass
                        
                        case 'time':
                            game_state = GameState(game_state.bankroll, float(message['time']), game_state.round_num)
                        
                        case 'info':
                            info = message['info']
                            seat = int(info['seat'])
                            new_game = 'new_game' in info and info['new_game']
                            last_nonterminal_state = round_state if isinstance(round_state, RoundState) or round_state is None else round_state.previous_state
                            
                            if new_game:
                                starting_stack = info['starting_stack'] if 'starting_stack' in info else STARTING_STACK
                                round_state = RoundState(
                                    turn_number = 0,
                                    street = 0,
                                    player_to_act = 0,
                                    pips = info['pips'] if 'pips' in info else [SMALL_BLIND, BIG_BLIND],
                                    stacks = info['stacks'] if 'stacks' in info else [starting_stack - SMALL_BLIND, starting_stack - BIG_BLIND],
                                    hands = info['hands'] if 'hands' in info else [None, None],
                                    deck = info['community'] if 'community' in info else [],
                                    n_ranks = info['n_ranks'] if 'n_ranks' in info else 13,
                                    n_streets = info['n_streets'] if 'n_streets' in info else 4,
                                    action_history = [],
                                    previous_state = None,
                                )
                            else:
                                round_state = RoundState(
                                    turn_number = last_nonterminal_state.turn_number,
                                    street = last_nonterminal_state.street,
                                    player_to_act = last_nonterminal_state.player_to_act,
                                    **{k: info[k] if k in info else getattr(last_nonterminal_state, k) for k in ['pips', 'stacks', 'hands', ]},
                                    deck = info['community'] if 'community' in info else last_nonterminal_state.deck,
                                    **{k: info[k] if k in info else getattr(last_nonterminal_state, k) for k in ['n_ranks', 'n_streets', ]},
                                    action_history = last_nonterminal_state.action_history,
                                    previous_state = round_state,
                                )
                            
                            if new_game:
                                self.pokerbot.handle_new_round(game_state, round_state, seat)
                        
                        case 'action':
                            if message['action']['verb'] in DECODE:
                                if message['action']['verb'] == 'Raise':
                                    (min_raise, max_raise) = round_state.raise_bounds()
                                    if min_raise <= message['action']['size'] and message['action']['size'] <= max_raise:
                                        action = RaiseAction(message['action']['size'])
                                    else:
                                        print(f'WARN Bad raise size from game server: {message} but raise bounds are {round_state.raise_bounds()}')
                                        action = CallAction()
                                else:
                                    action = DECODE[message['action']['verb']]()
                            else:
                                print(f'WARN Bad action type: {message}')
                            round_state = round_state.proceed(action)
                        
                        case 'payoff':
                            delta = message['payoff']
                            deltas = [-delta, -delta]
                            deltas[seat] = delta
                            round_state = TerminalState(deltas, round_state)
                            game_state = GameState(game_state.bankroll + delta, game_state.game_clock, game_state.round_num)
                            self.pokerbot.handle_round_over(game_state, round_state, seat)
                        
                        case 'goodbye':
                            return
                    
                        case _:
                            print(f"WARN Bad message type: {message}")
                        
                except KeyError as e:
                    print(f'WARN Message missing required field "{e}": {message}')
                    continue
            # if not round_state.player_to_act == seat:
            #     print(round_state)
            #     print(f'seat={seat}')
            assert round_state.player_to_act == seat
            action = self.pokerbot.get_action(game_state, round_state, seat)
            self.send(action, seat)

def parse_args():
    '''
    Parses arguments corresponding to socket connection information.
    '''
    parser = argparse.ArgumentParser(prog='python3 player.py')
    parser.add_argument('--host', type=str, default='localhost', help='Host to connect to, defaults to localhost')
    parser.add_argument('port', type=int, help='Port on host to connect to')
    return parser.parse_args()

def run_bot(pokerbot, args):
    '''
    Runs the pokerbot.
    '''
    assert isinstance(pokerbot, Bot)
    try:
        sock = socket.create_connection((args.host, args.port))
    except OSError:
        print('Could not connect to {}:{}'.format(args.host, args.port))
        return
    socketfile = sock.makefile('rw')
    runner = Runner(pokerbot, socketfile)
    runner.run()
    socketfile.close()
    sock.close()
