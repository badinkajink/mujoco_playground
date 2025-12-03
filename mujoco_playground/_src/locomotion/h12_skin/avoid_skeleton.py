"""Reference-tracking collision avoidance for H1-2 with skin sensors.

This environment trains an RL policy to track MPC-generated reference trajectories
while learning obstacle avoidance behaviors through capacitive skin sensing.

Architecture:
- Asymmetric Actor-Critic (actor sees noisy sensors, critic sees privileged state)
- Loads reference trajectories from MPC rollouts
- Computes capacitance-based proximity sensing
- Rewards trajectory tracking + collision avoidance
"""

from typing import Any, Dict, Optional, Tuple, Union
import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
import numpy as np
from pathlib import Path

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.h12_skin import base as h12_skin_base
from mujoco_playground._src.locomotion.h12_skin import h12_skin_constants as consts


def default_config() -> config_dict.ConfigDict:
  """Configuration for the avoidance task.
  
  Key parameters:
  - ctrl_dt: Policy control frequency (e.g., 0.02 = 50Hz)
  - sim_dt: Physics simulation timestep (e.g., 0.004 = 250Hz)
  - episode_length: Max steps per episode
  - history_len: Number of past observations to stack
  - obs_noise: Noise parameters for sensor readings
  - reward_config: Scales for different reward components
  - traj_dir: Directory containing MPC reference trajectories
  - smaller network sizes (e.g., [256, 256, 128, 128])
  """
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.004,
      episode_length=1000,
      early_termination=True,
      action_repeat=1,
      action_scale=0.6,
      history_len=10,  # N=10 for 50Hz history from 200Hz control (every 4 steps)
      obs_noise=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.01,
              joint_vel=1.5,
              gyro=0.2,
              gravity=0.05,
              capacitance=0.1,  # Noise on capacitance readings
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Task rewards (tracking)
              joint_pos_tracking=1.0,
              joint_vel_tracking=0.5,
              base_vel_tracking=1.0,
              base_angvel_tracking=0.5,
              torque_tracking=0.1,
              
              # Avoidance rewards
              collision_penalty=-100.0,  # Large penalty for actual collision
              proximity_penalty=-2.0,    # Penalty based on capacitance
              clearance_reward=1.0,      # Reward for maintaining safe distance
              
              # Regularization costs
              action_rate=-0.1,
              torque=-0.01,
              energy=-0.001,
              joint_limit=-1.0,
          ),
      ),
      capacitance_config=config_dict.create(
          eps=1.0,
          sensing_radius=0.15,
          collision_threshold=10.0,  # Capacitance value indicating collision
      ),
      traj_dir="./traj_logs",  # Directory with MPC trajectories
      impl="mjx",
      nconmax=8 * 8192,
      njmax=19 + 8 * 4,
  )


# -----------------------------------------------------------
# Capacitance Computation Functions (JAX-compatible)
# -----------------------------------------------------------

def compute_capacitance(
    sensor_pos: jax.Array,
    obst_pos: jax.Array,
    obst_radius: float,
    eps: float = 1.0,
    sensing_radius: float = 0.15
) -> jax.Array:
  """Compute capacitance-based proximity for a single sensor-obstacle pair.
  
  Pseudocode:
  1. Compute Euclidean distance d = ||sensor_pos - obst_pos||
  2. If d > sensing_radius + obst_radius: return -1 (out of range)
  3. effective_d = max(0.01, d - obst_radius)  # Surface distance
  4. Return eps / effective_d  # Inverse distance (higher = closer)
  
  Args:
    sensor_pos: (3,) sensor position
    obst_pos: (3,) obstacle position
    obst_radius: obstacle radius
    eps: capacitance constant
    sensing_radius: max sensing distance
  
  Returns:
    capacitance value (scalar), -1 if out of range
  """
  # TODO: Implement JAX version of C++ ComputeCapacitancePair
  # Use jp.linalg.norm, jp.where for conditionals
  pass


def compute_all_capacitances(
    data: mjx.Data,
    skin_site_ids: jax.Array,
    obstacle_body_id: int,
    obstacle_geom_id: int,
    model: mjx.Model,
    eps: float = 1.0,
    sensing_radius: float = 0.15
) -> jax.Array:
  """Compute capacitance for all skin sensors.
  
  Pseudocode:
  1. Get obstacle position from data.xpos[obstacle_body_id]
  2. Get obstacle radius from model.geom_size[obstacle_geom_id]
  3. For each sensor site in skin_site_ids:
       - Get sensor position from data.site_xpos[site_id]
       - Compute capacitance using compute_capacitance()
  4. Return array of shape (n_sensors,) with all capacitances
  
  Args:
    data: MJX physics state
    skin_site_ids: (n_sensors,) array of site IDs
    obstacle_body_id: ID of obstacle body
    obstacle_geom_id: ID of obstacle geometry
    model: MJX model
    eps: capacitance constant
    sensing_radius: max sensing distance
  
  Returns:
    (n_sensors,) array of capacitance values
  """
  # TODO: Use jax.vmap to vectorize over sensors
  # Handle case where obstacle doesn't exist (return zeros)
  pass


def get_skin_site_ids(mj_model) -> np.ndarray:
  """Extract all site IDs with 'sensor' in their name.
  
  Pseudocode:
  1. Iterate through model.nsite
  2. Check if site name contains 'sensor'
  3. Collect matching site IDs
  4. Return as numpy array
  
  This runs once at initialization (not in training loop).
  """
  # TODO: Implement site name search
  pass


# -----------------------------------------------------------
# Reference Trajectory Loading and Sampling
# -----------------------------------------------------------

class TrajectoryDatabase:
  """Manages loading and sampling of MPC reference trajectories.
  
  The database loads CSV files from traj_logs/ directory.
  Each CSV contains:
  - step, time
  - obst_px, obst_py, obst_pz (obstacle position)
  - obst_vx, obst_vy, obst_vz (obstacle velocity)
  - obst_radius
  - qpos_7, ..., qpos_33 (robot joint positions, excluding floating base and obstacle)
  - qvel_6, ..., qvel_32 (robot joint velocities, excluding floating base and obstacle)
  - cap_site_0_sensor_*, ... (capacitance readings for all 63 sensors)
  - cmd_vx, cmd_vy, cmd_vz (commanded obstacle velocity)
  """
  
  def __init__(self, traj_dir: str):
    """Load all trajectory files from directory.
    
    Pseudocode:
    1. Find all CSV files in traj_dir
    2. Load each CSV into memory
    3. Store as list of dictionaries with numpy arrays
    4. Each trajectory dict contains:
       - qpos: (T, 27) joint positions (7-33, excluding base and obstacle)
       - qvel: (T, 27) joint velocities (6-32)
       - obstacle_pos: (T, 3)
       - obstacle_vel: (T, 3)
       - obstacle_radius: (T,)
       - capacitances: (T, 63)
       - time: (T,)
    """
    # TODO: Implement CSV loading with pandas/numpy
    pass
  
  def sample_trajectory(self, rng: jax.Array) -> Dict[str, jax.Array]:
    """Randomly sample one reference trajectory.
    
    Pseudocode:
    1. Use jax.random to pick random trajectory index
    2. Convert selected trajectory to JAX arrays
    3. Return trajectory dict
    
    Returns:
      Dictionary with trajectory data (all as JAX arrays)
    """
    # TODO: Implement random sampling
    pass
  
  def get_reference_at_step(
      self,
      trajectory: Dict[str, jax.Array],
      step: int
  ) -> Dict[str, jax.Array]:
    """Extract reference values at specific timestep.
    
    Pseudocode:
    1. Clamp step to valid range [0, T-1]
    2. Index into trajectory arrays at step
    3. Return dict with scalar/vector references
    
    Returns:
      ref_qpos: (27,)
      ref_qvel: (27,)
      ref_obstacle_pos: (3,)
      ref_obstacle_vel: (3,)
      ref_capacitances: (63,)
    """
    # TODO: Implement timestep indexing with bounds checking
    pass


# -----------------------------------------------------------
# Main Environment Class
# -----------------------------------------------------------

class Avoid(h12_skin_base.H12SkinEnv):
  """Collision avoidance environment with reference trajectory tracking."""
  
  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    """Initialize environment.
    
    Pseudocode:
    1. Call parent __init__ with XML path
    2. Load trajectory database
    3. Initialize obstacle body/geom IDs
    4. Initialize skin sensor site IDs
    5. Set up observation/action spaces
    6. Initialize default poses and control ranges
    """
    super().__init__(
        xml_path=consts.FEET_ONLY_XML.as_posix(),  # TODO: Update to h12_skin XML
        config=config,
        config_overrides=config_overrides,
    )
    self._config = config
    self._post_init()
  
  def _post_init(self) -> None:
    """Post-initialization setup.
    
    Pseudocode:
    1. Load trajectory database from config.traj_dir
    2. Extract obstacle body/geom IDs from model
    3. Get skin sensor site IDs (63 sensors)
    4. Store default pose from keyframe
    5. Store actuator control ranges
    6. Define which joints to track (hx_idxs, weights)
    7. Store PD gains if needed
    """
    # TODO: Initialize trajectory database
    # self._traj_db = TrajectoryDatabase(self._config.traj_dir)
    
    # TODO: Get obstacle IDs
    # self._obstacle_body_id = ...
    # self._obstacle_geom_id = ...
    
    # TODO: Get skin sensor IDs
    # self._skin_site_ids = jp.array(get_skin_site_ids(self._mj_model))
    
    # TODO: Store default pose and control ranges
    # self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    # self._default_pose = self._init_q[7:34]  # Joints 7-33 (27 actuated)
    # self._lowers = self._mj_model.actuator_ctrlrange[:, 0]
    # self._uppers = self._mj_model.actuator_ctrlrange[:, 1]
    pass
  
  def reset(self, rng: jax.Array) -> mjx_env.State:
    """Reset environment and sample new reference trajectory.
    
    Pseudocode:
    1. Sample random reference trajectory from database
    2. Initialize robot to starting pose (could be from trajectory or default)
    3. Initialize obstacle to starting position from trajectory
    4. Reset history buffers (qpos error, qvel, actions, capacitances)
    5. Compute initial observations (actor and critic)
    6. Initialize metrics dict
    7. Return initial State
    
    Key info dict fields to initialize:
    - rng: random key
    - current_trajectory: sampled reference trajectory
    - traj_step: current step in trajectory (starts at 0)
    - last_act: zero actions
    - qpos_error_history: (history_len * 27,)
    - qvel_history: (history_len * 27,)
    - action_history: (history_len * 27,)
    - capacitance_history: (history_len * 63,)
    - motor_targets: (27,)
    """
    # TODO: Implement full reset logic
    pass
  
  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    """Step environment forward.
    
    Pseudocode:
    1. Apply action (with scaling and clipping)
    2. Step physics for n_substeps
    3. Increment trajectory step counter
    4. Compute capacitances for current state
    5. Get reference values for current step
    6. Update history buffers
    7. Compute observations (actor and critic)
    8. Compute rewards (tracking + avoidance)
    9. Check termination conditions
    10. Update state and return
    
    Args:
      state: current state
      action: (27,) action offsets from default pose
    
    Returns:
      new state with updated data, obs, reward, done, metrics, info
    """
    # TODO: Implement step logic
    pass
  
  def _get_obs(
      self,
      data: mjx.Data,
      info: Dict[str, Any],
      rng: jax.Array,
  ) -> jax.Array:
    """Construct observation for actor.
    
    Actor observation (noisy proprioception + partial reference):
    - Noisy proprioception (sapt):
      * gyro (3)
      * gravity (3)
      * joint_pos - default_pose (27)
      * joint_vel (27)
      * qpos_error_history (history_len * 27)
      * qvel_history (history_len * 27)
      * action_history (history_len * 27)
      * capacitance readings (63) [NOISY]
      * capacitance_history (history_len * 63)
    
    - Partial reference (sart):
      * ref_joint_vel (27)
      * ref_base_angvel (3)
      * ref_joint_pos (27)
      
    Total size: ~(3+3+27+27 + 10*27 + 10*27 + 10*27 + 63 + 10*63 + 27+3+27) = ~1500+
    
    Pseudocode:
    1. Get base proprioception (gyro, gravity, joint pos/vel)
    2. Add noise to proprioception based on config.obs_noise
    3. Get capacitance readings and add noise
    4. Get history buffers from info
    5. Get reference values for current step
    6. Concatenate all components
    7. Return full observation vector
    """
    # TODO: Implement observation construction with noise
    pass
  
  def _get_critic_obs(
      self,
      data: mjx.Data,
      info: Dict[str, Any],
  ) -> jax.Array:
    """Construct privileged observation for critic.
    
    Critic observation (privileged state + full reference):
    - Privileged proprioception (scpt):
      * base_pos (3)
      * base_quat (4)
      * base_linvel (3)
      * base_angvel (3)
      * gravity (3)
      * joint_pos_history (history_len * 27)
      * joint_vel (27)
      * action_history (history_len * 27)
      * TRUE capacitances (63) [NO NOISE]
      * capacitance_history (history_len * 63)
      * obstacle position (3)
      * obstacle velocity (3)
      * PD gains Kp, Kd (if using)
    
    - Full reference state (scrt):
      * ref_base_pos (3)
      * ref_base_quat (4)
      * ref_base_linvel (3)
      * ref_base_angvel (3)
      * ref_joint_pos (27)
      * ref_joint_vel (27)
      * ref_capacitances (63)
      
    Pseudocode:
    1. Get full privileged state (no noise)
    2. Get true capacitances (no noise)
    3. Get obstacle state from data
    4. Get full reference trajectory info
    5. Concatenate all components
    6. Return critic observation
    """
    # TODO: Implement critic observation (privileged)
    pass
  
  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: Dict[str, Any],
      ref_state: Dict[str, jax.Array],
  ) -> Tuple[Dict[str, jax.Array], Dict[str, jax.Array]]:
    """Compute reward components.
    
    Returns positive rewards and negative costs separately.
    Final reward = sum(pos_rewards * scales) * exp(0.2 * sum(neg_costs * scales)) * dt
    
    Positive rewards (tracking):
    - joint_pos_tracking: exp(-||qpos - ref_qpos||^2 / sigma)
    - joint_vel_tracking: exp(-||qvel - ref_qvel||^2 / sigma)
    - base_vel_tracking: exp(-||base_vel - ref_base_vel||^2 / sigma)
    - base_angvel_tracking: exp(-||angvel - ref_angvel||^2 / sigma)
    - torque_tracking: exp(-||torque - ref_torque||^2 / sigma)
    - clearance_reward: reward for keeping distance > threshold
    
    Negative costs (regularization + penalties):
    - collision_penalty: -100 * any(capacitance > threshold)
    - proximity_penalty: -sum(capacitance^2) where capacitance > 0
    - action_rate: -||action - last_action||^2
    - torque: -||torque||^2
    - energy: -||torque * joint_vel||
    - joint_limit: -penalty for approaching joint limits
    
    Pseudocode:
    1. Get current state (qpos, qvel, torques, etc.)
    2. Get reference state for tracking
    3. Compute tracking errors
    4. Compute exponential tracking rewards
    5. Get capacitance readings
    6. Compute collision/proximity penalties
    7. Compute regularization costs
    8. Return two dicts: pos_rewards, neg_costs
    """
    # TODO: Implement all reward components from MPC residuals
    pass
  
  def _get_termination(self, data: mjx.Data, capacitances: jax.Array) -> jax.Array:
    """Check termination conditions.
    
    Terminate if:
    1. Joint limits exceeded
    2. Robot falls (gravity_z < threshold)
    3. Collision detected (any capacitance > threshold)
    4. Robot position too far from starting area
    
    Pseudocode:
    1. Check joint angles against limits
    2. Check torso orientation (gravity sensor)
    3. Check capacitances for collision
    4. Check robot hasn't moved too far
    5. Return boolean (or float 0/1)
    """
    # TODO: Implement termination checks
    pass
  
  # -----------------------------------------------------------
  # Helper reward functions (matching MPC residuals)
  # -----------------------------------------------------------
  
  def _reward_joint_tracking(
      self,
      qpos: jax.Array,
      ref_qpos: jax.Array,
      weights: Optional[jax.Array] = None
  ) -> jax.Array:
    """Exponential reward for joint position tracking.
    
    Pseudocode:
    1. Compute error = qpos - ref_qpos
    2. If weights provided, apply per-joint weights
    3. Return exp(-weighted_error^2 / sigma)
    """
    # TODO: Implement with exponential form
    pass
  
  def _cost_proximity(self, capacitances: jax.Array) -> jax.Array:
    """Penalize proximity to obstacle via capacitance.
    
    Pseudocode:
    1. Filter capacitances > 0 (in sensing range)
    2. Compute sum of capacitance^2 (inverse distance squared)
    3. Return negative cost
    """
    # TODO: Implement proximity cost
    pass
  
  def _reward_clearance(
      self,
      capacitances: jax.Array,
      safe_threshold: float = 2.0
  ) -> jax.Array:
    """Reward for maintaining safe clearance.
    
    Pseudocode:
    1. Count sensors with capacitance < safe_threshold
    2. Reward proportional to number of safe sensors
    3. Return positive reward
    """
    # TODO: Implement clearance reward
    pass
  
  def _cost_collision(
      self,
      capacitances: jax.Array,
      collision_threshold: float = 10.0
  ) -> jax.Array:
    """Large penalty for collision.
    
    Pseudocode:
    1. Check if any(capacitance > collision_threshold)
    2. Return large negative penalty if true, 0 otherwise
    """
    # TODO: Implement collision detection and penalty
    pass
  
  def _cost_action_rate(
      self,
      action: jax.Array,
      last_action: jax.Array
  ) -> jax.Array:
    """Penalize rapid action changes.
    
    Pseudocode:
    1. Compute delta = action - last_action
    2. Return -||delta||^2
    """
    # TODO: Implement action smoothness cost
    pass
  
  def _cost_energy(
      self,
      torques: jax.Array,
      qvel: jax.Array
  ) -> jax.Array:
    """Penalize energy consumption.
    
    Pseudocode:
    1. Compute power = torques * qvel
    2. Return -sum(abs(power))
    """
    # TODO: Implement energy cost
    pass


# -----------------------------------------------------------
# Additional Notes
# -----------------------------------------------------------

"""
Implementation checklist:

1. Trajectory loading:
   - Parse CSV files from MPC rollouts
   - Handle variable trajectory lengths
   - Efficient random sampling with JAX RNG

2. Capacitance sensing:
   - JAX-compatible compute_capacitance function
   - Vectorized computation over all 63 sensors
   - Handle obstacle radius from model

3. Observations:
   - Actor: noisy sensors + partial reference
   - Critic: privileged state + full reference
   - History buffers with proper rolling
   - Noise injection with proper scales

4. Rewards:
   - Map MPC residuals to RL rewards
   - Exponential tracking rewards
   - Collision/proximity penalties
   - Regularization costs
   - Proper scaling and combination

5. Termination:
   - Joint limits, falls, collisions
   - Early termination flag in config

6. Domain randomization (future):
   - Observation noise curriculum
   - Dynamics randomization
   - External force perturbations
   - Delayed actions

7. Integration:
   - Compatible with MuJoCo Playground training loop
   - Works with PPO/SAC algorithms
   - Proper State object management
"""
