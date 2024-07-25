'''
Poker Camp Game Engine
(c) 2024 poker.camp; all rights reserved.

Derived from: 6.176 MIT Pokerbots Game Engine at mitpokerbots/engine
'''
import argparse
import traceback
from collections import namedtuple
from threading import Thread
from queue import Queue
import time
import json
from pathlib import Path
import subprocess
import socket
# import eval7
import sys
import os

sys.path.append(os.getcwd())
from config import *

import random

random.seed(68127)

class LeducDeck:
    def __init__(self):
        self.cards = [0, 1, 2, 0, 1, 2]
        random.shuffle(self.cards)
        self.index = 0
    
    def deal(self):
        if self.index >= len(self.cards):
            raise ValueError("No more cards to deal")
        card = self.cards[self.index]
        self.index += 1
        return card
    
    def peek(self, street):
        if street >= len(self.cards):
            raise ValueError("Index out of range")
        return self.cards[2:street+2]

class FoldAction(namedtuple('FoldAction', [])):
    def __hash__(self):
        return hash('FoldAction')
    def __eq__(self, other):
        return isinstance(other, FoldAction)
    def __repr__(self):
        return 'Fold'
    
class CallAction(namedtuple('CallAction', [])):
    def __hash__(self):
        return hash('CallAction')
    def __eq__(self, other):
        return isinstance(other, CallAction)
    def __repr__(self):
        return 'Call'
    
class CheckAction(namedtuple('CheckAction', [])):
    def __hash__(self):
        return hash('CheckAction')
    def __eq__(self, other):
        return isinstance(other, CheckAction)
    def __repr__(self):
        return 'Check'
    
class RaiseAction(namedtuple('RaiseAction', [])):
    def __hash__(self):
        return hash('RaiseAction')
    def __eq__(self, other):
        return isinstance(other, RaiseAction)
    def __repr__(self):
        return 'Raise'


TerminalState = namedtuple('TerminalState', ['deltas', 'previous_state'])

STREET_NAMES = ['Flop']
DECODE = {'F': FoldAction, 'C': CallAction, 'K': CheckAction, 'R': RaiseAction}
CCARDS = lambda card: '{}'.format(card)
PCARDS = lambda card: '[{}]'.format(card)
PVALUE = lambda name, value: f', {name} ({value:+d})'
STATUS = lambda players: ''.join([PVALUE(p.name, p.bankroll) for p in players])

# A socket line may be a MESSAGE or a list of MESSAGEs. You are expected
# to respond once to every newline.
#
# A MESSAGE is a json object with a 'type' field (STRING). Depending on
#   the type, zero or more other fields may be required:
# hello -> [no fields]
# time -> time : FLOAT match timer remaining
# info -> info : INFO_DICT information available to you
# action -> action : ACTION_DICT, player : INT
# payoff -> payoff : FLOAT incremental payoff to you
# goodbye -> [no fields]
#
# An INFO_DICT may include game-dependent fields; most games will have:
# seat : INT your player number
# new_game : optional BOOL, set to true if this message starts a new game
# 
# An ACTION_DICT includes a 'verb' field (STRING). Depending on the
#   game, and possibly the verb, additional fields may be required or
#   optional.
# ACTION_DICTs sent by the server always include a 'seat' field (INT)
#   which identifies which player acted. This field is optional for
#   messages you send.
# 
# In the course of a round, you should receive:
# info message with new_game = true
# one or more action messages or info messages
# a payoff message
# The actions report both players' actions (including yours) in order.
#
# A player takes an action by sending a legal action message. This is
#  currently the only legal message for players to send.

def message(type, **kwargs):
    result = {'type': type}
    result.update(kwargs)
    return result

class RoundState(namedtuple('_RoundState', ['turn_number', 'street', 'pips', 'stacks', 'hands', 'deck', 'action_history', 'previous_state'])):
    @staticmethod
    def new():
        '''
        Returns a RoundState representing the start of a Leduc game.
        '''
        deck = LeducDeck()
        hands = [deck.deal(), deck.deal()]
        pips = [ANTE, ANTE]
        stacks = [STARTING_STACK - ANTE, STARTING_STACK - ANTE]
        return RoundState(
            turn_number = 0,
            street = 0,
            pips = pips,
            stacks = stacks,
            hands = hands,
            deck = deck,
            action_history = [],
            previous_state = None,
        )
    
    def showdown(self):
        assert self.pips[0] == self.pips[1]
        
        hands = self.hands
        street = self.street
        assert(street == 1)
        community = self.deck.peek(street)

        if hands[0] == community:
            winner = 0
        elif hands[1] == community:
            winner = 1
        elif hands[0] > hands[1]:
            winner = 0
        elif hands[1] > hands[0]:
            winner = 1
        else:
            winner = None #Tie 

        deltas = [0, 0]
        if winner is not None:
            loser = 1 - winner
            deltas[winner] = self.pips[loser]
            deltas[loser] = -self.pips[loser]
        
        return TerminalState(deltas, self)
    
    def visible_hands(self, seat):
        ret = [None for _ in self.hands]
        ret[seat] = self.hands[seat]
        return ret

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
                action_history = self.action_history + [CheckAction()],
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

class Player():
    '''
    Handles subprocess and socket interactions with one player's pokerbot.
    '''

    def __init__(self, name, path, output_dir):
        self.name = name
        self.path = path
        self.stdout_path = f'{output_dir}/{self.name}.stdout.txt'
        self.game_clock = STARTING_GAME_CLOCK
        self.bankroll = 0
        self.commands = None
        self.bot_subprocess = None
        self.messages = [message('time', time=30.)]
        self.socketfile = None
        self.bytes_queue = Queue()
        self.message_log = []
        self.response_log = []

    def build(self):
        '''
        Loads the commands file and builds the pokerbot.
        '''
        try:
            with open(self.path + '/commands.json', 'r') as json_file:
                commands = json.load(json_file)
            if ('build' in commands and 'run' in commands and
                    isinstance(commands['build'], list) and
                    isinstance(commands['run'], list)):
                self.commands = commands
            else:
                print(self.name, 'commands.json missing command')
        except FileNotFoundError:
            print(self.name, 'commands.json not found - check PLAYER_PATH')
        except json.decoder.JSONDecodeError:
            print(self.name, 'commands.json misformatted')
        if self.commands is not None and len(self.commands['build']) > 0:
            try:
                proc = subprocess.run(self.commands['build'],
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      cwd=self.path, timeout=BUILD_TIMEOUT, check=False)
                self.bytes_queue.put(proc.stdout)
            except subprocess.TimeoutExpired as timeout_expired:
                error_message = 'Timed out waiting for ' + self.name + ' to build'
                print(error_message)
                self.bytes_queue.put(timeout_expired.stdout)
                self.bytes_queue.put(error_message.encode())
            except (TypeError, ValueError):
                print(self.name, 'build command misformatted')
            except OSError:
                print(self.name, 'build failed - check "build" in commands.json')

    def run(self):
        '''
        Runs the pokerbot and establishes the socket connection.
        '''
        if self.commands is not None and len(self.commands['run']) > 0:
            try:
                server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                with server_socket:
                    server_socket.bind(('', 0))
                    server_socket.settimeout(CONNECT_TIMEOUT)
                    server_socket.listen()
                    port = server_socket.getsockname()[1]
                    proc = subprocess.Popen(
                        self.commands['run'] + [str(port)],
                        # stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        cwd=self.path,
                    )
                    self.bot_subprocess = proc
                    # function for bot listening
                    def enqueue_output(out, queue):
                        try:
                            for line in out:
                                queue.put(line)
                        except ValueError:
                            pass
                    # start a separate bot listening thread which dies with the program
                    Thread(target=enqueue_output, args=(proc.stdout, self.bytes_queue), daemon=True).start()
                    # block until we timeout or the player connects
                    client_socket, _ = server_socket.accept()
                    with client_socket:
                        client_socket.settimeout(CONNECT_TIMEOUT)
                        sock = client_socket.makefile('rw')
                        self.socketfile = sock
                        self.append(message('hello'))
                        print(self.name, 'connected successfully')
            except (TypeError, ValueError):
                print(self.name, 'run command misformatted')
            except OSError:
                print(self.name, 'run failed - check "run" in commands.json')
            except socket.timeout:
                print('Timed out waiting for', self.name, 'to connect')

    def stop(self, as_player):
        '''
        Closes the socket connection and stops the pokerbot.
        '''
        if self.socketfile is not None:
            try:
                self.append(message('goodbye'))
                self.query(None, None, wait=False)
                self.socketfile.close()

            except socket.timeout:
                print('Timed out waiting for', self.name, 'to disconnect')

            except OSError:
                print('Could not close socket connection with', self.name)
       
        if self.bot_subprocess is not None:
            try:
                outs, _ = self.bot_subprocess.communicate(timeout=CONNECT_TIMEOUT)
                self.bytes_queue.put(outs)
            except subprocess.TimeoutExpired:
                print(f'Timed out waiting for {self.name} to quit')
                self.bot_subprocess.kill()
                outs, _ = self.bot_subprocess.communicate()
                self.bytes_queue.put(outs)
        with open(self.stdout_path, 'wb') as log_file:
            bytes_written = 0
            for output in self.bytes_queue.queue:
                try:
                    bytes_written += log_file.write(output)
                    if bytes_written >= PLAYER_LOG_SIZE_LIMIT:
                        break
                except TypeError:
                    pass

    def append(self, msg):
        self.messages.append(msg)

    def query(self, round_state, game_log, *, wait=True):
        legal_actions = round_state.legal_actions() if isinstance(round_state, RoundState) else set()
        if self.socketfile is not None and (self.game_clock > 0. or not wait):
            clause = ''
            try:
                self.messages[0] = message(
                    'time',
                    time = round(self.game_clock, 3),
                )
                packet = json.dumps(self.messages)
                del self.messages[1:]  # do not duplicate messages
                self.message_log.append(packet)
                start_time = time.perf_counter()
                self.socketfile.write(packet + '\n')
                self.socketfile.flush()
                if not wait:
                    return None
                response = self.socketfile.readline().strip()
                # print(f"Raw response from {self.name}: {response}")  # Debug print
                # if not response:
                #     raise ValueError("Empty response from player")
                end_time = time.perf_counter()
                self.response_log.append(response)
                if ENFORCE_GAME_CLOCK:
                    self.game_clock -= end_time - start_time
                if self.game_clock <= 0.:
                    raise socket.timeout
                
                action = None
                
                if len(response) > 0 and (response[0] == '[' or response[0] == '{'):
                    # okay to accept a single json object
                    if response[0] == '{':
                        response = f'[{response}]'
                    for response_obj in json.loads(response):
                        try:
                            if response_obj['type'] == 'action':
                                action_verb = response_obj['action']['verb']
                                if action_verb == 'F':
                                    action = FoldAction()
                                elif action_verb == 'C':
                                    action = CallAction()
                                elif action_verb == 'K':
                                    action = CheckAction()
                                elif action_verb == 'R':
                                    action = RaiseAction()
                                    # if 'amount' in response_obj['action']:
                                    #     action = RaiseAction(response_obj['action']['amount'])
                                    # else:
                                    #     raise ValueError("Raise action must specify an amount")
                                else:
                                    raise ValueError(f'Invalid action verb: {action_verb}')
                            else:
                                raise ValueError(f"Invalid message type: {response_obj['type']}")
                        except KeyError as e:
                            print(f'WARN Message missing required field "{e}": {response_obj}')
                            continue
                else:
                    print(f'WARN Bad message format (expected json or list of json): {response}')
                    print(f'WARN Bad message format (expected json or list of json): {response}')
                
                if action in legal_actions:
                    return action()
                print('response obj: ', response_obj)
                print('action: ', action)
                game_log.append(f'{self.name} attempted illegal {action.__repr__()}')
            except socket.timeout:
                error_message = f'{self.name} ran out of time'
                game_log.append(error_message)
                print(error_message)
                self.game_clock = 0.
            except OSError:
                error_message = self.name + ' disconnected'
                game_log.append(error_message)
                print(error_message)
                self.game_clock = 0.
            except (IndexError, KeyError, ValueError):
                game_log.append(self.name + ' response misformatted: ' + str(clause))
        return CheckAction() if CheckAction in legal_actions else FoldAction()

class Match():
    '''
    Manages logging and the high-level game procedure.
    '''

    def __init__(self, p1, p2, output_path, n_rounds):
        global PLAYER_1_NAME, PLAYER_1_PATH, PLAYER_2_NAME, LOGS_PATH, PLAYER_2_PATH, NUM_ROUNDS
        if p1 is not None:
            PLAYER_1_NAME = p1[0]
            PLAYER_1_PATH = p1[1]
        if p2 is not None:
            PLAYER_2_NAME = p2[0]
            PLAYER_2_PATH = p2[1]
        if output_path is not None:
            LOGS_PATH = output_path
        if n_rounds is not None:
            NUM_ROUNDS = int(n_rounds)
        
        self.log = ['Poker Camp Game Engine - ' + PLAYER_1_NAME + ' vs ' + PLAYER_2_NAME]

    def send_round_state(self, players, round_state):
    def send_round_state(self, players, round_state):
        '''
        Incorporates RoundState information into the game log and player messages.

        No-op if there's no new information to share.
        '''
        if round_state.street == 0 and round_state.turn_number == 0:
            for seat, player in enumerate(players):
                self.log.append(f'{player.name} posts the ante of {ANTE}')
            for seat, player in enumerate(players):
                self.log.append(f'{player.name} dealt {PCARDS(round_state.hands[seat])}')
                player.append(message('info', info={
                    'seat': seat,
                    'hands': round_state.visible_hands(seat),
                    'new_game': True,
                }))
        elif round_state.street == 1 and round_state.turn_number == 0:
            board = round_state.deck.peek(round_state.street)
            self.log.append(
                STREET_NAMES[round_state.street - 1]
                + ' '
                + PCARDS(board)
                + ''.join(
                    PVALUE(player.name, round_state.pips[seat])
                    for seat, player in enumerate(players)
                )
            )
            for seat, player in enumerate(players):
                player.append(message('info', info={
                    'seat': seat,
                    'hands': round_state.visible_hands(seat),
                    'board': board,
                }))

    def send_action(self, action, *, players, seat, round_state, ):
        '''
        Incorporates action information into the game log and player messages.
        '''
        if isinstance(action, FoldAction):
            phrasing = ' folds'
            code = 'F'
        elif isinstance(action, CallAction):
            phrasing = ' calls'
            code = 'C'
        elif isinstance(action, CheckAction):
            phrasing = ' checks'
            code = 'K'
        elif isinstance(action, RaiseAction):
            phrasing = f' raises {round_state.raise_bounds()}'
            code = f'R'
        else:
            raise ValueError(f"Unknown action type: {type(action)}")

        self.log.append(players[seat].name + phrasing)
        for player in players:
            player.append(message(
                'action',
                action =
                    {'verb': code, 'amount': round_state.raise_bounds()}
                    if isinstance(action, RaiseAction)
                    else {'verb': code}
                ,
                seat = seat,
            ))

    def send_terminal_state(self, players, round_state):
        '''
        Incorporates TerminalState information into the game log and player messages.
        '''
        previous_state = round_state.previous_state

        if previous_state.pips[0] == previous_state.pips[1]:
            for seat, player in enumerate(players):
                self.log.append(f'{player.name} shows {PCARDS(previous_state.hands[seat])}')
                player.append(message('info', info = {
                    'seat': seat,
                    'hands': previous_state.hands,
                })) # we could put a board here if street == 1, but meh
        
        for seat, player in enumerate(players):
            self.log.append(f'{player.name} awarded {round_state.deltas[seat]:+d}')
            player.append(message(
                'payoff',
                payoff = round_state.deltas[seat],
            ))

    def run_round(self, players):
        round_state = RoundState.new()
        while not isinstance(round_state, TerminalState):
            self.send_round_state(players, round_state)
            active = round_state.turn_number % 2
            action = players[active].query(
                round_state,
                self.log,
            )
            self.send_action(action, players=players, seat=active, round_state=round_state)
            round_state = round_state.proceed(action)
        
        self.send_terminal_state(players, round_state)
        for player, delta in zip(players, round_state.deltas):
            player.bankroll += delta

    def run(self):
        '''
        Runs one match of poker.
        '''
        print('Starting the game engine...')
        MATCH_DIR = f'{LOGS_PATH}/{PLAYER_1_NAME}.{PLAYER_2_NAME}'
        Path(MATCH_DIR).mkdir(parents=True, exist_ok=True)
        players = [
            Player(PLAYER_1_NAME, PLAYER_1_PATH, MATCH_DIR, ),
            Player(PLAYER_2_NAME, PLAYER_2_PATH, MATCH_DIR, ),
        ]
        for player in players:
            player.build()
            player.run()
        for round_num in range(1, NUM_ROUNDS + 1):
            self.log.append('')
            self.log.append('Round #' + str(round_num) + STATUS(players))
            self.run_round(players)
            players = players[::-1]
        self.log.append('')
        self.log.append('Final' + STATUS(players))
        for i, player in enumerate(players):
            player.stop(i)
            
        print('Writing logs...')
        
        with open(f'{MATCH_DIR}/{GAME_LOG_FILENAME}.txt', 'w') as log_file:
            log_file.write('\n'.join(self.log))
        for active, name in [
            (0, PLAYER_1_NAME),
            (1, PLAYER_2_NAME),
        ]:
            with open(f'{MATCH_DIR}/{name}.msg.server.txt', 'w') as log_file:
                log_file.write('\n'.join(players[active].message_log))
            with open(f'{MATCH_DIR}/{name}.msg.player.txt', 'w') as log_file:
                log_file.write('\n'.join(players[active].response_log))
        
        with open(f'{LOGS_PATH}/{SCORE_FILENAME}.{PLAYER_1_NAME}.{PLAYER_2_NAME}.txt', 'w') as score_file:
            score_file.write('\n'.join([f'{p.name},{p.bankroll*100.0/NUM_ROUNDS}' for p in players]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Game engine with optional player arguments")
    parser.add_argument('-p1', nargs=2, metavar=('NAME', 'FILE'), help='Name and executable for player 1')
    parser.add_argument('-p2', nargs=2, metavar=('NAME', 'FILE'), help='Name and executable for player 2')
    parser.add_argument("-o", "--output", required=True, default="logs", help="Output directory for game results")
    parser.add_argument("-n", "--n_rounds", default=1000, help="Number of rounds to run per matchup")

    args = parser.parse_args()
    
    Match(args.p1, args.p2, args.output, args.n_rounds).run()
