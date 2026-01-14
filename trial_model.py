def build_rlgym_v2_env():
    from rlgym.api import RLGym
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition, TimeoutCondition, AnyCondition
    from rlgym.rocket_league.reward_functions import CombinedReward
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import MutatorSequence, KickoffMutator
    from rlgym_ppo.util.rlgym_v2_gym_wrapper import RLGymV2GymWrapper
    from rlgym_tools.rocket_league.renderers.rocketsimvis_renderer import RocketSimVisRenderer
    from rlgym_tools.rocket_league.state_mutators.variable_team_size_mutator import VariableTeamSizeMutator
    from rlgym_tools.rocket_league.state_mutators.random_physics_mutator import RandomPhysicsMutator

    from rlgym.rocket_league.obs_builders import DefaultObs
    from rewards.common_rewards import FaceBallReward
    no_touch_timeout_seconds = 30
    game_timeout_seconds = 10

    action_parser = RepeatAction(LookupTableAction(), repeats=8)
                    
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=no_touch_timeout_seconds),
        TimeoutCondition(timeout_seconds=game_timeout_seconds),
        GoalCondition(),
    )
    termination_condition = AnyCondition(
        TimeoutCondition(timeout_seconds=game_timeout_seconds)
    )
    
    reward_fn = CombinedReward(
        (FaceBallReward(), 0.002),
    )

    obs_builder = DefaultObs()
    
    mode_weights = {
        (1, 1): 0.33,  # 33% chance of 1v1
    }   
    state_mutator = MutatorSequence(  
        VariableTeamSizeMutator(mode_weights=mode_weights),
        KickoffMutator()
    )

    rlgym_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        renderer=RocketSimVisRenderer(udp_ip="100.90.35.22"),
        shared_info_provider=None,
    )

    return RLGymV2GymWrapper(rlgym_env)

from rocket_learn.agent.discrete_ff import DiscreteFF
def load_models(model_paths, device):
    models = []
    for path in model_paths:
        model = DiscreteFF(172, 90, [256, 256, 256], device)
        model.load_state_dict(torch.load(path)['actor_state_dict'])
        model.eval()
        models.append(model)
    return models


if __name__ == '__main__':
    import torch
    import numpy as np
    from torch.distributions import Categorical
    import time

    device = torch.device("cuda")
    ts = 8
    env = build_rlgym_v2_env()
    blue_score = 0
    orange_score = 0
    max_goals = 100
    deterministic = False
    m = 1
    model_paths =  ["checkpoint_save_directory/demo_latest\demo_-1\checkpoint.pt"] * 2
    try:
        with torch.no_grad():
            models = load_models(model_paths, device)
            print("Starting Evaluation")

            # Initialize timing vars
            start_time = time.perf_counter()
            tick_duration = ts / 120
            next_tick_time = start_time + tick_duration
            
            while True:
                obs = env.reset()
                state = env.rlgym_env.state
                gamemode = len(state.cars) // 2
                done = False
                k = 0
                prev_actions = None
                # hidden = [None for _ in range(len(state.cars))]
                while not done:
                    outputs = []
                    for i in range(2 * gamemode):
                        model_idx = i + 3 * (i // (2 * gamemode))
                        input_tensor = torch.from_numpy(np.expand_dims(obs[i][:173], 0)).float().to(device)
                        output = models[model_idx](input_tensor)

                        # Apply different temperatures based on model index
                        temperature = 1 if model_idx == 0 else 1.0
                        scaled_output = output / temperature
                        outputs.append(scaled_output)
                    out = torch.cat(outputs)
                    dist = Categorical(logits=out)
                    action_indices = out.argmax(dim=-1).cpu().numpy() if deterministic else dist.sample().cpu().numpy()

                    action_dict = {}
                    actions_array = []
                    for i in range(2 * gamemode):
                        action_idx = action_indices[i]
                        action = np.array([action_idx], dtype=np.int32)
                        action_dict[env.agent_map[i]] = action
                        actions_array.append(action[0])

                    # if (ts * k) % 120 == 0:
                    #     print(action_indices, dist.entropy().mean().item())

                    for _ in range(ts // ts):
                        obs, reward, done, truncated, info = env.step(np.array(actions_array, dtype=np.int32).reshape(-1, 1))
                        done = done or truncated
                        state = env.rlgym_env.state
                        env.rlgym_env.render()
                        if state.goal_scored:
                            if state.scoring_team == 0:
                                blue_score += 1
                            else:
                                orange_score += 1
                            print(f"Score: {blue_score} - {orange_score}", end="\r", flush=True)
                        
                        # Synchronize TPS using perf_counter
                        current_time = time.perf_counter()
                        time_to_next_tick = next_tick_time - current_time
                        if time_to_next_tick > 0:
                            time.sleep(time_to_next_tick)
                        
                        # Schedule the next tick
                        next_tick_time += tick_duration
                        
                        if done:
                            break
                    k += 1
                if blue_score >= max_goals or orange_score >= max_goals:
                    break
            print(blue_score - orange_score)
    finally:
        env.close()
