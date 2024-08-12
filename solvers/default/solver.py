'''
Example solver. Implements a kind of crude Counterfactual Regret Minimization.
'''
import argparse
import eval7

from skeleton.base_solver import BaseSolver
from skeleton.engine import RoundState, CallAction, RaiseAction
from skeleton.runner import parse_args, run_solver

class Solver(BaseSolver):
    def __init__(self, n_ranks, n_streets, starting_stack):
        self.n_ranks = n_ranks
        self.n_streets = n_streets
        self.starting_stack = starting_stack
        self.regrets = {}
        self.strategy_sums = {}
        self.infoset_visit_count = {}
    
    def handle_new_iteration(self, _):
        '''
        Called when iteration is about to begin.
        '''
        
        pass
    
    def handle_iteration_over(self, _):
        '''
        Called when iteration is over.
        '''
        pass
    
    def get_root(self, _):
        '''
        Returns the RoundState to start a new iteration at.
        '''
        
        return RoundState.new(
            n_ranks=self.n_ranks,
            n_streets=self.n_streets,
            starting_stack=self.starting_stack,
        )
    
    def determine_infoset(self, state):
        '''
        Called to ask for the canonical name of an infoset.
        
        You might use this to coalesce multiple gamehistories into the same
        infoset.
        '''
        
        hand = state.hands[state.player_to_act]
        suited = (hand[0].suit == hand[1].suit)
        hand = ''.join([eval7.ranks[i] for i in sorted([card.rank for card in hand], reverse=True)])
        
        if state.street == 0:
            return f'P{state.player_to_act+1}:{state.street}:{hand}{"s" if suited else "o"}{state.action_history}'
        
        return f'P{state.player_to_act+1}:{state.street}:{hand}{state.public()["community"]}{state.action_history}'
    
    def sample_actions(self, infoset, legal_actions, raise_bounds):
        '''
        legal_actions is a list of types; this function should give one or more
        instances of each type (it might be 'or more' if you need to try
        different bet sizes...)
        '''
        
        raise_sizes = []
        
        if raise_bounds[1] > 0:
            size = raise_bounds[0]
            while size <= raise_bounds[1] and len(raise_sizes) < 10:
                raise_sizes.append(int(size))
                size *= 1.5
            if raise_sizes[-1] != raise_bounds[1]:
                raise_sizes.append(raise_bounds[1])
        return (
            [action() for action in legal_actions if action != RaiseAction]
            + ([RaiseAction(size) for size in raise_sizes] if RaiseAction in legal_actions else [])
        )
    
    def get_sampling_policy(self, infoset, *, iteration):
        '''
        Called to ask for the sampling policy at this infoset.
        
        You can let this vary by iteration #, if you like.
        Currently supported: "expand_all", "sample"
        '''
        
        if (infoset[0:2] == 'P1') ^ (iteration % 2 == 0):
            return 'sample'
        else:
            return 'expand_all'
    
    def handle_new_samples(
        self,
        infoset,
        sampling_policy,
        samples,
        use_infoset_ev,
    ):
        '''
        Called on each new set of samples produced for a node.
        
        You should use this opportunity to update your strategy probabilites
        (or whatever stored values they are derived from, like cumulative
        regrets)
        '''
        
        if sampling_policy == 'expand_all':
            for action, sample_list in samples.items():
                regret = sample_list[0] - use_infoset_ev
                
                if infoset not in self.regrets:
                    self.regrets[infoset] = {}
                    
                if action not in self.regrets[infoset]:
                    self.regrets[infoset][action] = 0
                
                # if regret > 0:
                #     print(f'add {max(regret, 0)} to {action} at {infoset}; EV {sample_list[0]} > {use_ev}')
                self.regrets[infoset][action] = max(0, self.regrets[infoset][action] + regret)
    
    def get_training_strategy_probabilities(self, infoset, legal_actions):
        '''
        Called to ask for the current probabilities that should be used in a
        training iteration.
        '''
        
        total_regret = sum(self.regrets[infoset].values()) if infoset in self.regrets else None
        if infoset not in self.regrets or total_regret == 0:
            n_legal_actions = len(legal_actions)
            return {action : 1 / n_legal_actions for action in legal_actions}
        probabilities = {
            action : value / total_regret
            for action, value
            in self.regrets[infoset].items()
        }
        
        if infoset not in self.strategy_sums:
            self.strategy_sums[infoset] = {action: 0 for action in legal_actions}
            self.infoset_visit_count[infoset] = 0
        
        self.infoset_visit_count[infoset] += 1
        
        if self.infoset_visit_count[infoset] > 5:
            for action, prob in probabilities.items():
                self.strategy_sums[infoset][action] += prob
            
        return probabilities
    
    def get_final_strategy_probabilities(self):
        '''
        Called to ask for the final probabilities to report.
        
        This returns the average of all training probabilities for each infoset.
        '''
        final_probabilities = {}
        
        for infoset, action_sums in self.strategy_sums.items():
            visit_count = self.infoset_visit_count[infoset] - 5
            final_probabilities[infoset] = {
                action: sum_prob / visit_count
                for action, sum_prob in action_sums.items()
                if visit_count > 0
            }
        
        return final_probabilities

# usage: python3 solver.py --iter ITERATIONS
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the poker solver")
    parser.add_argument('--iter', type=int, required=True, help="Number of iterations")
    parser.add_argument('--ranks', type=int, default=5, metavar='INT', help="Number of ranks")
    parser.add_argument('--streets', type=int, default=2, metavar='INT', help="Number of streets")
    parser.add_argument('--starting-stack', type=int, default=20, metavar='INT', help="Starting stack size")
    
    args = parser.parse_args()
    
    run_solver(
        Solver(
            n_ranks=args.ranks,
            n_streets=args.streets,
            starting_stack=args.starting_stack,
        ),
        args,
    )
