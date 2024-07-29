"""
Simple example pokerbot, written in Python.
"""

import random

from skeleton.actions import CallAction, CheckAction, FoldAction, RaiseAction
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot
from skeleton.states import (
    ANTE,
    BET_SIZE,
    NUM_ROUNDS,
    STARTING_STACK,
    GameState,
    RoundState,
    TerminalState,
)


class Player(Bot):
    """
    A pokerbot.
    """

    def __init__(self):
        """
        Called when a new game starts. Called exactly once.

        Arguments:
        Nothing.

        Returns:
        Nothing.
        """
        # with (Path(__file__).parent / "strategy.json").open("r") as f:
        #     self.strategy = json.load(f)
        #     print(f"Loaded strategy: {self.strategy}")
        self.strategy = {
            "Q_": 0.0,
            "K_": 0.0,
            "A_": 0.0,
            "_QD": 1.0 / 3.0,
            "_KD": 0.0,
            "_AD": 1.0,
            "_QU": 0.0,
            "_KU": 1.0 / 3.0,
            "_AU": 1.0,
            "Q_DU": 0.0,
            "K_DU": 1.0 / 3.0,
            "A_DU": 1.0,
        }

    def handle_new_round(self, game_state, round_state, active):
        """
        Called when a new round starts. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Nothing.
        """
        # my_bankroll = game_state.bankroll  # the total number of chips you've gained or lost from the beginning of the game to the start of this round
        # game_clock = game_state.game_clock  # the total number of seconds your bot has left to play this game
        # round_num = game_state.round_num  # the round number from 1 to NUM_ROUNDS
        # my_cards = round_state.hands[active]  # your cards
        # big_blind = bool(active)  # True if you are the big blind
        pass

    def handle_round_over(self, game_state, terminal_state, active):
        """
        Called when a round ends. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        terminal_state: the TerminalState object.
        active: your player's index.

        Returns:
        Nothing.
        """
        # my_delta = terminal_state.deltas[active]  # your bankroll change from this round
        # previous_state = terminal_state.previous_state  # RoundState before payoffs
        # street = previous_state.street  # 0, 3, 4, or 5 representing when this round ended
        # my_cards = previous_state.hands[active]  # your cards
        # opp_cards = previous_state.hands[1-active]  # opponent's cards or [] if not revealed
        pass

    def get_action(self, game_state, round_state, active):
        """
        Where the magic happens - your code should implement this function.
        Called any time the engine needs an action from your bot.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Your action.
        """
        legal_actions = (
            round_state.legal_actions()
            if isinstance(round_state, RoundState)
            else set()
        )  # the actions you are allowed to take
        street = (
            round_state.street
        )  # 0, 3, 4, or 5 representing pre-flop, flop, turn, or river respectively
        my_cards = round_state.hands[active]  # your cards
        if street > 0:
            board_cards = round_state.deck[:street][0]  # the board cards
        # my_pip = round_state.pips[active]  # the number of chips you have contributed to the pot this round of betting
        # opp_pip = round_state.pips[1-active]  # the number of chips your opponent has contributed to the pot this round of betting
        # my_stack = round_state.stacks[active]  # the number of chips you have remaining
        # opp_stack = round_state.stacks[1-active]  # the number of chips your opponent has remaining
        # continue_cost = opp_pip - my_pip  # the number of chips needed to stay in the pot
        # my_contribution = STARTING_STACK - my_stack  # the number of chips you have contributed to the pot
        # opp_contribution = STARTING_STACK - opp_stack  # the number of chips your opponent has contributed to the pot
        # if RaiseAction in legal_actions:
        #    min_raise, max_raise = round_state.raise_bounds()  # the smallest and largest numbers of chips for a legal bet/raise
        #    min_cost = min_raise - my_pip  # the cost of a minimum bet/raise
        #    max_cost = max_raise - my_pip  # the cost of a maximum bet/raise

        # print(round_state)

        # manual strategy
        if street == 0:
            if my_cards == 2:
                if RaiseAction in legal_actions and random.random() < 1.0 / 2.0:
                    return RaiseAction()
            elif my_cards == 0:
                if RaiseAction in legal_actions and random.random() < 1.0 / 5.0:
                    return RaiseAction()

            if CheckAction in legal_actions:
                return CheckAction()
            else:
                return CallAction()
        elif street == 1:
            remaining_cards = [c for c in [0, 1, 2] if c not in board_cards]
            # print(
            #     f"cards {my_cards}, board {board_cards[0]}, remaining {remaining_cards}"
            # )
            if my_cards == board_cards[0]:
                # A in QKA game
                if RaiseAction in legal_actions:
                    return RaiseAction()
                elif CallAction in legal_actions:
                    return CallAction()
                else:
                    print(
                        f"WARN: couldn't raise nuts with {my_cards}, board {board_cards}"
                    )
            elif my_cards == remaining_cards[1]:
                # K in QKA game
                if CheckAction in legal_actions:
                    return CheckAction()
                elif CallAction in legal_actions:
                    return CallAction()
            elif my_cards == remaining_cards[0]:
                # Q in QKA game
                if RaiseAction in legal_actions and random.random() < 1.0 / 5.0:
                    return RaiseAction()
                elif CheckAction in legal_actions:
                    return CheckAction()
                else:
                    return FoldAction()
            else:
                print(f"WARN: couldn't parse hand with {my_cards}, board {board_cards}")
        else:
            print(f"WARN: unexpected street {street}")

        print(f"WARN: unexpected fallback action")
        if CheckAction in legal_actions:
            return CheckAction()
        else:
            return FoldAction()


if __name__ == "__main__":
    run_bot(Player(), parse_args())
