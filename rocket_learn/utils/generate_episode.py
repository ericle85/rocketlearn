from typing import List

import numpy as np
import torch

from rocket_learn.agent.pretrained_policy import HardcodedAgent
from rocket_learn.utils.env_wrapper import RLGymV2GymWrapper
from tqdm import tqdm

from rocket_learn.agent.policy import Policy
from rocket_learn.experience_buffer import ExperienceBuffer
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator
from rlgym_tools.rocket_league.state_mutators.random_scoreboard_mutator import RandomScoreboardMutator

def generate_episode(env: RLGymV2GymWrapper, policies, evaluate=False, scoreboard=None, progress=False) -> tuple[list[ExperienceBuffer], int]:
    """
    create experience buffer data by interacting with the environment(s)
    """
    if progress:
        progress = tqdm(unit=" steps")
    else:
        progress = None

    if evaluate:  # Change setup temporarily to play a normal game (approximately)
        from rlgym.rocket_league.sim import RocketSimEngine
        from rlgym.rocket_league.reward_functions.goal_reward import GoalReward
        from rlgym.rocket_league.done_conditions import NoTouchTimeoutCondition, AnyCondition

        from rlgym_tools.rocket_league.state_mutators.game_mutator import GameMutator
        from eval_game_utils import GameCondition, ScoreboardInitMutator, KickoffOnGoalEngine

        # Saving training env settings so we can reset back to them after eval
        og_state_mutator = env.rlgym_env.state_mutator
        og_reward_fn = env.rlgym_env.reward_fn
        og_truncation_condition = env.rlgym_env.truncation_cond
        og_termination_condition = env.rlgym_env.termination_cond
        og_engine = env.rlgym_env.transition_engine

        eval_transition_engine = KickoffOnGoalEngine(RocketSimEngine())
        eval_state_mutator = MutatorSequence(
            FixedTeamSizeMutator(),
            ScoreboardInitMutator(),
            GameMutator(),
        )
        eval_termination_condition = AnyCondition(
            GameCondition(seconds_per_goal_forfeit=10 * len(env.rlgym_env.agents), max_overtime_seconds=300),
        )
        eval_truncation_condition = NoTouchTimeoutCondition(timeout_seconds=60)

        env.rlgym_env.transition_engine = eval_transition_engine
        env.rlgym_env.state_mutator = eval_state_mutator
        env.rlgym_env.termination_cond = eval_termination_condition
        env.rlgym_env.truncation_cond = eval_truncation_condition
        env.rlgym_env.reward_fn = GoalReward()  # noqa

    if scoreboard is not None:
        og_state_mutator = env.rlgym_env.state_mutator
        if not evaluate:
            env.rlgym_env.state_mutator = MutatorSequence(
                RandomScoreboardMutator(),
                env.rlgym_env.state_mutator
            )
    observations = env.reset()
    last_state = env.rlgym_env.state
    result = 0

    latest_policy_indices = [0 if isinstance(p, HardcodedAgent) else 1 for p in policies]
    # rollouts for all latest_policies
    rollouts = [
        ExperienceBuffer(infos=[])
        for _ in range(sum(latest_policy_indices))
    ]

    b = o = 0
    with torch.no_grad():
        while True:
            all_indices = []
            all_actions = []
            all_log_probs = []

            if not isinstance(policies[0], HardcodedAgent) and all(policy == policies[0] for policy in policies):
                policy = policies[0]
                if isinstance(observations[0], tuple):
                    obs = tuple(np.concatenate([obs[i] for obs in observations], axis=0)
                                for i in range(len(observations[0])))
                else:
                    obs = observations
                dist = policy.get_action_distribution(obs)
                action_indices = policy.sample_action(dist)
                log_probs = policy.log_prob(dist, action_indices)
                actions = policy.env_compatible(action_indices)

                all_indices.extend(list(action_indices.numpy()))
                all_actions.extend(list(actions))
                all_log_probs.extend(list(log_probs.numpy()))
            else:
                index = 0
                for policy, obs in zip(policies, observations):
                    if isinstance(policy, HardcodedAgent):
                        actions = policy.act(last_state, index)

                        # make sure output is in correct format
                        if not isinstance(observations, np.ndarray):
                            actions = np.array(actions)

                        # TODO: add converter that takes normal 8 actions into action space
                        # actions = env._match._action_parser.convert_to_action_space(actions)

                        all_indices.append(None)
                        all_actions.append(actions)
                        all_log_probs.append(None)

                    elif isinstance(policy, Policy):
                        dist = policy.get_action_distribution(obs)
                        action_indices = policy.sample_action(dist)[0]
                        log_probs = policy.log_prob(dist, action_indices).item()
                        actions = policy.env_compatible(action_indices)

                        all_indices.append(action_indices.numpy())
                        all_actions.append(actions)
                        all_log_probs.append(log_probs)

                    else:
                        print(str(type(policy)) + " type use not defined")
                        assert False

                    index += 1

            # to allow different action spaces, pad out short ones to the longest length (assume later unpadding in parser)
            # length = max([a.shape[0] for a in all_actions])
            # padded_actions = []
            # for a in all_actions:
            #     action = np.pad(a.astype('float64'), (0, length - a.size), 'constant', constant_values=np.NAN)
            #     padded_actions.append(action)
            #
            # all_actions = padded_actions
            # TEST OUT ABOVE TO DEAL WITH VARIABLE LENGTH

            all_actions = np.vstack(all_actions)
            old_obs = observations
            observations, rewards, terminated, truncated, info = env.step(all_actions)

            if len(policies) <= 1:
                observations, rewards = [observations], [rewards]

            # prune data that belongs to old agents
            old_obs = [a for i, a in enumerate(old_obs) if latest_policy_indices[i] == 1]
            all_indices = [d for i, d in enumerate(all_indices) if latest_policy_indices[i] == 1]
            rewards = [r for i, r in enumerate(rewards) if latest_policy_indices[i] == 1]
            all_log_probs = [r for i, r in enumerate(all_log_probs) if latest_policy_indices[i] == 1]

            # Create lists for terminated and truncated to match other variables
            terminated_list = [terminated] * len(old_obs)
            truncated_list = [truncated] * len(old_obs)

            assert len(old_obs) == len(all_indices), str(len(old_obs)) + " obs, " + str(len(all_indices)) + " ind"
            assert len(old_obs) == len(rewards), str(len(old_obs)) + " obs, " + str(len(rewards)) + " ind"
            assert len(old_obs) == len(all_log_probs), str(len(old_obs)) + " obs, " + str(len(all_log_probs)) + " ind"
            assert len(old_obs) == len(rollouts), str(len(old_obs)) + " obs, " + str(len(rollouts)) + " ind"

            # Might be different if only one agent?
            if not evaluate:  # Evaluation matches can be long, no reason to keep them in memory
                for exp_buf, obs, act, rew, term, trunc, log_prob in zip(rollouts, old_obs, all_indices, rewards,
                                                                         terminated_list, truncated_list, all_log_probs):
                    exp_buf.add_step(obs, act, rew, term, trunc, log_prob, info)

            if progress is not None:
                progress.update()
                igt = progress.n * 4 / 120  # TODO: Instead of hardcoding TS (4) maybe use the scoreboard?

                prog_str = f"{igt // 60:02.0f}:{igt % 60:02.0f} IGT"
                if evaluate:
                    prog_str += f", BLUE {b} - {o} ORANGE"
                progress.set_postfix_str(prog_str)

            if env.rlgym_env.state.goal_scored:
                if env.rlgym_env.state.scoring_team == 0:
                    b += 1
                else:
                    o += 1

            if terminated or truncated:
                break

            last_state = info['state']

    #TODO: Everything below here
    if scoreboard is not None:
        env.rlgym_env.state_mutator = og_state_mutator

    if evaluate:
        env.rlgym_env.transition_engine = og_engine
        env.rlgym_env.state_mutator = og_state_mutator
        env.rlgym_env.termination_condition = og_termination_condition
        env.rlgym_env.truncation_cond = og_truncation_condition
        env.rlgym_env.reward_fn = og_reward_fn
        return result

    if progress is not None:
        progress.close()

    return rollouts, result
