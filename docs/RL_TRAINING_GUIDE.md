# RL Training Guide for Air65 Drone

## Quick Start

### 1. Environment Setup

```bash
# Activate conda environment
conda activate betafpv-ultrathink

# Verify installations
python -c "import gymnasium, torch, pybullet; print('All packages installed!')"
```

### 2. Complete Skeleton Implementations

The following files need completion (marked with TODO):

1. **`src/rl/drone_sim.py`** - PyBullet physics simulator
   - Priority: HIGH
   - Estimated time: 4-6 hours
   - Key: Implement quadrotor dynamics and motor mixing

2. **`src/rl/drone_env.py`** - Gymnasium environment
   - Priority: HIGH
   - Estimated time: 2-3 hours
   - Key: Complete reset() and step() methods

3. **`src/agents/puffer_policy.py`** - Neural network policy
   - Priority: MEDIUM
   - Estimated time: 2 hours
   - Key: Build Actor-Critic architecture

4. **`src/agents/puffer_trainer.py`** - Training loop
   - Priority: MEDIUM
   - Estimated time: 3-4 hours
   - Key: Implement PPO update and PufferLib integration

5. **`src/agents/policy_runner.py`** - Inference wrapper
   - Priority: LOW
   - Estimated time: 1 hour
   - Key: Load checkpoint and run policy

### 3. Training Workflow

Once implementations are complete:

```bash
# Train hover stabilization
python scripts/train.py

# Monitor training
tensorboard --logdir data/logs

# Evaluate trained policy
python scripts/evaluate.py --checkpoint data/checkpoints/best_model.pt

# Visualize in simulation
python scripts/visualize_policy.py --checkpoint data/checkpoints/best_model.pt
```

## Implementation Order

### Phase 1: Simulator (Week 1)

**File**: `src/rl/drone_sim.py`

**Steps**:
1. Complete PyBullet initialization in `__init__()`
2. Implement `reset()` - reset drone pose and velocities
3. Implement `_action_to_motor_commands()` - control mixing
4. Implement `step()` - apply forces, step physics
5. Implement `_get_state()` - extract position, velocity, attitude
6. Test with random actions

**Testing**:
```python
from src.rl.drone_sim import DroneSimulator
sim = DroneSimulator(gui=True)  # Visual debugging
state = sim.reset()
for _ in range(1000):
    action = np.random.uniform(-1, 1, 4)
    state = sim.step(action)
    time.sleep(0.01)
```

### Phase 2: Environment (Week 2)

**File**: `src/rl/drone_env.py`

**Steps**:
1. Load configs in `__init__()`
2. Initialize `SimDroneInterface` with your simulator
3. Implement `reset()` - reset backend, normalize observation
4. Implement `step()` - execute action, compute reward, check termination
5. Test with Gymnasium's `env.check` utility

**Testing**:
```python
from src.rl.drone_env import DroneHoverEnv
env = DroneHoverEnv()
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, trunc, info = env.step(action)
    if done or trunc:
        obs, info = env.reset()
```

### Phase 3: Policy Network (Week 2)

**File**: `src/agents/puffer_policy.py`

**Steps**:
1. Build encoder layers in `__init__()`
2. Build actor and critic heads
3. Implement `forward()` - pass through network
4. Implement `get_action()` - sample from policy
5. Test forward pass with random input

### Phase 4: Training Loop (Week 3)

**File**: `src/agents/puffer_trainer.py`

**Steps**:
1. Setup PufferLib vectorized environments
2. Implement rollout collection
3. Implement GAE advantage computation
4. Implement PPO update
5. Add logging and checkpointing

## Configuration

### Hyperparameter Tuning

Key hyperparameters in `configs/training_config.yaml`:

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `learning_rate` | 3e-4 | [1e-5, 1e-3] | Higher = faster learning, less stable |
| `gamma` | 0.99 | [0.95, 0.999] | Discount factor for future rewards |
| `clip_range` | 0.2 | [0.1, 0.3] | PPO clipping - higher = more aggressive updates |
| `ent_coef` | 0.01 | [0.0, 0.1] | Exploration bonus |
| `num_envs` | 64 | [16, 128] | Parallel environments - more = faster |

### Reward Shaping

Edit `configs/env_config.yaml`:

```yaml
rewards:
  position_weight: 10.0    # Increase for tighter position control
  velocity_weight: 5.0     # Increase to penalize oscillations
  attitude_weight: 3.0     # Increase for more level flight
  energy_weight: -0.001    # Increase (less negative) for efficiency
```

## Troubleshooting

### Issue: Drone crashes immediately

**Solution**:
- Check initial hover thrust calculation
- Verify motor mixing matrix signs
- Ensure action normalization is correct
- Try starting with zero initial velocity

### Issue: Learning doesn't progress

**Checks**:
- Is reward signal informative? (should vary based on actions)
- Are observations normalized? (critical for neural networks)
- Is action space exploration sufficient? (check entropy in logs)
- Try reducing learning rate or increasing batch size

### Issue: Sim-trained policy fails on real drone

**Solutions**:
- Increase domain randomization in simulation
- Add noise to observations during training
- Fine-tune on real hardware (scripts/sim_to_real.py)
- Check for simulator biases (ground effect, drag, etc.)

## Performance Benchmarks

Expected training progress:

| Timesteps | Success Rate | Avg Episode Reward | Notes |
|-----------|--------------|-------------------|-------|
| 100k | ~10% | -20 | Random exploration |
| 500k | ~50% | 5 | Basic hovering |
| 1M | ~80% | 15 | Stable hover |
| 2M+ | ~95% | 20+ | Production-ready |

Training time on different hardware:
- **CPU only**: ~12 hours for 1M steps (64 envs)
- **GPU (RTX 3080)**: ~3 hours for 1M steps
- **Apple Silicon (M1/M2)**: ~6 hours for 1M steps

## Next Steps After Hover Mastery

1. **Waypoint Navigation**: Change `task_type: "waypoint"` in env_config.yaml
2. **Trajectory Following**: Try figure-8 or circular paths
3. **Vision-Based Control**: Add camera observations to state
4. **Multi-Drone Formation**: Train coordinated policies
5. **Acrobatic Maneuvers**: Flips, rolls with specialized rewards

## Resources

- **PPO Paper**: https://arxiv.org/abs/1707.06347
- **Gymnasium Docs**: https://gymnasium.farama.org/
- **PufferLib**: https://github.com/PufferAI/PufferLib
- **PyBullet Guide**: https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet
- **RL Baselines Zoo**: https://github.com/DLR-RM/rl-baselines3-zoo (for hyperparameters)

## Safety Guidelines

When deploying to real hardware:

1. ✅ Always remove propellers for initial testing
2. ✅ Test in a netted/enclosed area
3. ✅ Have emergency stop accessible
4. ✅ Start with 50% max throttle limit
5. ✅ Monitor battery voltage continuously
6. ✅ Use tether for first free flights
7. ✅ Have fire extinguisher nearby (LiPo safety)

## Support

For questions or issues:
1. Check `CLAUDE.md` for architectural overview
2. Review configuration files in `configs/`
3. Examine skeleton implementations for structure
4. Reference existing working code (`state.py`, `rewards.py`, etc.)

Happy training! 🚁
