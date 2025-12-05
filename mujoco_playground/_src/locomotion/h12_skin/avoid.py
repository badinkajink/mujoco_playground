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
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import warnings

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.h12_skin import base as h12_skin_base
from mujoco_playground._src.locomotion.h12_skin import h12_skin_constants as consts


def default_config() -> config_dict.ConfigDict:
  """Configuration for the avoidance task."""
  return config_dict.create(
      ctrl_dt=0.01,
      sim_dt=0.002,
      episode_length=300,  # Long enough for trajectory + recovery
      early_termination=True,
      action_repeat=1,
      action_scale=0.6,
      history_len=3,
      
      # Add PD gains if needed (like Go1)
      Kp=100.0,  # Adjust for your robot
      Kd=2.0,
      
      obs_noise=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.01,
              joint_vel=1.5,
              gyro=0.2,
              gravity=0.05,
              capacitance=0.1,
          ),
      ),
      
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Tracking rewards (active during MPC trajectory)
              joint_pos_tracking=1.0,
              joint_vel_tracking=0.5,
              base_vel_tracking=1.0,
              base_angvel_tracking=0.5,
              torque_tracking=0.0, # 0.1, i dont think this even works
              
              # Recovery rewards (active after trajectory ends)
              recovery_upright=1.0,
              recovery_height=1.0,
              recovery_posture=0.5,
              recovery_stability=0.5,
              recovery_feet_contact=0.0, # 0.3, seems superfluous
              
              # Avoidance rewards (always active)
              clearance_reward=0.0, #1.0, i dont think this is a good reward, zeroing out
              
              # Penalties (always active)
              collision_penalty=-100.0,
              proximity_penalty=-2.0,
              
              # Regularization (always active)
              action_rate=-0.1,
              torque=-0.01,
              energy=-0.001,
              joint_limit=-1.0,
          ),
      ),
      
      capacitance_config=config_dict.create(
          eps=1.0,
          sensing_radius=0.15,
          collision_threshold=80.0,
          safe_threshold=10.0, # determined ad hoc by looking at some csvs, 1/0.10m
      ),
      
      traj_dir="/home/wxie/workspace/h1_mujoco/augmented",
      impl="jax",
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
  
  Mirrors C++ ComputeCapacitancePair function.
  
  Args:
    sensor_pos: (3,) sensor position in world frame
    obst_pos: (3,) obstacle center position in world frame
    obst_radius: obstacle radius (scalar)
    eps: capacitance constant (default 1.0)
    sensing_radius: max sensing distance (default 0.15m)
  
  Returns:
    Capacitance value (scalar): eps/effective_distance if in range, -1.0 otherwise
  """
  # Compute Euclidean distance
  delta = sensor_pos - obst_pos
  d = jp.linalg.norm(delta)
  
  # Check if within sensing range
  in_range = d <= (sensing_radius + obst_radius)
  
  # Compute effective distance (distance to surface, not center)
  effective_d = jp.maximum(0.01, d - obst_radius)
  
  # Compute capacitance (inverse distance)
  capacitance = eps / effective_d
  
  # Return -1.0 if out of range, otherwise return capacitance
  return jp.where(in_range, capacitance, -1.0)


def compute_all_capacitances(
    data,  # mjx.Data type
    skin_site_ids: jax.Array,
    obstacle_body_id: int,
    obstacle_geom_id: int,
    model,  # mjx.Model type
    eps: float = 1.0,
    sensing_radius: float = 0.15
) -> jax.Array:
  """Compute capacitance for all skin sensors.
  
  Mirrors C++ ComputeAllCapacitances function.
  
  Args:
    data: MJX physics state
    skin_site_ids: (n_sensors,) array of site IDs (e.g., 63 sensors)
    obstacle_body_id: ID of obstacle body
    obstacle_geom_id: ID of obstacle geometry
    model: MJX model
    eps: capacitance constant
    sensing_radius: max sensing distance
  
  Returns:
    (n_sensors,) array of capacitance values
  """
  # Get obstacle position (body position)
  obst_pos = data.xpos[obstacle_body_id]
  
  # Get obstacle radius from geometry
  # geom_size is (ngeom, 3), first component is radius for sphere
  obst_radius = model.geom_size[obstacle_geom_id, 0]
  
  # Get all sensor positions
  # site_xpos is (nsite, 3), index with skin_site_ids
  sensor_positions = data.site_xpos[skin_site_ids]
  
  # Vectorized computation over all sensors using vmap
  compute_single = lambda sensor_pos: compute_capacitance(
      sensor_pos, obst_pos, obst_radius, eps, sensing_radius
  )
  
  capacitances = jax.vmap(compute_single)(sensor_positions)
  
  return capacitances


def get_skin_site_ids(mj_model) -> np.ndarray:
  """Extract all site IDs with 'sensor' in their name.
  
  This runs once at initialization (not in training loop).
  Returns numpy array (not JAX array) since it's static.
  
  Args:
    mj_model: mujoco.MjModel (not mjx.Model)
  
  Returns:
    numpy array of site IDs
  """
  site_ids = []
  for i in range(mj_model.nsite):
    site_name = mj_model.site(i).name
    if site_name is not None and "sensor" in site_name:
      site_ids.append(i)
  
  return np.array(site_ids, dtype=np.int32)


def compute_obstacle_centroid(
    data,  # mjx.Data
    skin_site_ids: jax.Array,
    capacitances: jax.Array
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Compute weighted centroid of detected obstacles.
  
  Mirrors the C++ obstacle centroid computation in avoid.cc.
  Used for computing directional avoidance rewards.
  
  Args:
    data: MJX physics state
    skin_site_ids: (n_sensors,) sensor site IDs
    capacitances: (n_sensors,) capacitance values
  
  Returns:
    obstacle_centroid: (3,) weighted centroid position
    total_weight: scalar sum of weights
    num_detections: scalar count of positive detections
  """
  # Get sensor positions
  sensor_positions = data.site_xpos[skin_site_ids]  # (n_sensors, 3)
  
  # Filter for positive detections (capacitance > 0 means in range)
  detected = capacitances > 0.0
  
  # Use capacitance as weight (higher = closer = more weight)
  weights = jp.where(detected, capacitances, 0.0)
  
  # Weighted sum of positions
  weighted_positions = sensor_positions * weights[:, None]  # (n_sensors, 3)
  weighted_sum = jp.sum(weighted_positions, axis=0)  # (3,)
  
  # Total weight and detection count
  total_weight = jp.sum(weights)
  num_detections = jp.sum(detected.astype(jp.float32))
  
  # Compute centroid (with safety check for zero weight)
  centroid = jp.where(
      total_weight > 0.0,
      weighted_sum / total_weight,
      jp.zeros(3)
  )
  
  return centroid, total_weight, num_detections


# Optional: Helper for debugging/visualization
def format_capacitance_readings(
    skin_site_ids: np.ndarray,
    capacitances: jax.Array,
    mj_model,
    threshold: float = 0.0
) -> str:
  """Format capacitance readings for logging/debugging.
  
  Args:
    skin_site_ids: numpy array of site IDs
    capacitances: JAX array of capacitance values
    mj_model: mujoco model (for site names)
    threshold: only show readings above this value
  
  Returns:
    Formatted string with sensor readings
  """
  cap_np = np.array(capacitances)
  lines = []
  for i, (sid, cap) in enumerate(zip(skin_site_ids, cap_np)):
    if cap > threshold:
      site_name = mj_model.site(sid).name
      lines.append(f"  {site_name} (id={sid}): {cap:.4f}")
  
  if not lines:
    return "  No detections above threshold"
  return "\n".join(lines)

# -----------------------------------------------------------
# Reference Trajectory Loading and Sampling
# -----------------------------------------------------------

class TrajectoryDatabase:
  """Manages loading and sampling of MPC reference trajectories.
  
  The database loads CSV files from traj_logs/ directory.
  Each CSV contains 500Hz data with:
  - step, time
  - obst_px, obst_py, obst_pz, obst_vx, obst_vy, obst_vz, obst_radius
  - qpos_0 to qpos_33 (34 values: 7 for floating base, 27 for joints, 0 unused for obstacle)
  - qvel_0 to qvel_32 (33 values: 6 for floating base, 27 for joints)
  - cap_site_0 to cap_site_63 (63 capacitance sensor readings)
  - cmd_vx, cmd_vy, cmd_vz (commanded obstacle velocity)
  
  Note: We only need robot joints (qpos 7-33, qvel 6-32) since floating base
  is simulated and obstacle state is tracked separately.
  """
  
  def __init__(self, traj_dir: str, downsample_factor: int = 1):
    """Load all trajectory files from directory.
    
    Args:
      traj_dir: Directory containing CSV trajectory files
      downsample_factor: Factor to downsample 500Hz to match control frequency
                        (e.g., 10 for 50Hz, 25 for 20Hz)
    """
    self.traj_dir = Path(traj_dir)
    self.downsample_factor = downsample_factor
    self.trajectories = []
    self.trajectory_lengths = []
    # Define controlled joint indices (21 joints)
    self.controlled_qpos_indices = [
        7, 8, 9, 10, 11, 12,     # Left leg
        13, 14, 15, 16, 17, 18,  # Right leg
        19,                       # Torso
        20, 21, 22, 23,           # Left arm (no wrist)
        27, 28, 29, 30,           # Right arm (no wrist)
    ]
    self.controlled_qvel_indices = [i - 1 for i in self.controlled_qpos_indices]  # qvel offset by -1

    if not self.traj_dir.exists():
      raise FileNotFoundError(f"Trajectory directory not found: {traj_dir}")
    
    # Find all CSV files
    csv_files = sorted(self.traj_dir.glob("*episode_*.csv"))
    
    if len(csv_files) == 0:
      raise ValueError(f"No CSV files found in {traj_dir}")
    
    print(f"Loading {len(csv_files)} trajectory files from {traj_dir}...")
    
    # Load each trajectory
    for csv_file in csv_files:
      try:
        traj = self._load_trajectory(csv_file)
        self.trajectories.append(traj)
        self.trajectory_lengths.append(traj['time'].shape[0])
      except Exception as e:
        warnings.warn(f"Failed to load {csv_file.name}: {e}")
        continue
    
    if len(self.trajectories) == 0:
      raise ValueError("No valid trajectories loaded")
    self.max_traj_len = max(traj['time'].shape[0] for traj in self.trajectories)
    self.num_trajectories = len(self.trajectories)
    # Stack all trajectories with padding
    def pad_to_max(arr, max_len):
        pad_len = max_len - arr.shape[0]
        if pad_len == 0:
            return arr
        # Repeat last element
        pad_arr = jp.repeat(arr[-1:], pad_len, axis=0)
        return jp.concatenate([arr, pad_arr], axis=0)
        
    # Convert to stacked JAX arrays (N_traj, max_len, ...)
    keys = self.trajectories[0].keys()
    self.stacked_trajectories = {}
    for key in keys:
      arrays = [pad_to_max(jp.array(t[key]), self.max_traj_len) 
                for t in self.trajectories]
      self.stacked_trajectories[key] = jp.stack(arrays)    
    # self.trajectory_lengths = np.array(self.trajectory_lengths)
    self.trajectory_lengths = jp.array([t['time'].shape[0] for t in self.trajectories])

    print(f"Successfully loaded {self.num_trajectories} trajectories")
    print(f"Trajectory lengths: min={self.trajectory_lengths.min()}, "
          f"max={self.trajectory_lengths.max()}, "
          f"mean={self.trajectory_lengths.mean():.1f}")
  
  def _load_trajectory(self, csv_file: Path) -> Dict[str, np.ndarray]:
    """Load a single trajectory CSV file.
    Args:
      csv_file: Path to CSV file
    Returns:
      Dictionary with trajectory data as numpy arrays
    """
    # Load CSV
    df = pd.read_csv(csv_file)
    
    # Downsample if needed (500Hz -> target frequency)
    if self.downsample_factor > 1:
        df = df.iloc[::self.downsample_factor].reset_index(drop=True)
    
    # Extract time
    time = df['time'].values
    
    # Extract obstacle state (freejoint body)
    obstacle_pos = df[['obst_px', 'obst_py', 'obst_pz']].values  # (T, 3)
    obstacle_vel = df[['obst_vx', 'obst_vy', 'obst_vz']].values  # (T, 3)
    obstacle_radius = df['obst_radius'].values  # (T,)
        
    # Extract robot joint positions (only 21 controlled joints)
    qpos_cols = [f'qpos_{i}' for i in self.controlled_qpos_indices]
    robot_qpos = df[qpos_cols].values  # (T, 21)
    
    # Extract robot joint velocities (only 21 controlled joints)
    qvel_cols = [f'qvel_{i}' for i in self.controlled_qvel_indices]
    robot_qvel = df[qvel_cols].values  # (T, 21)
    
    # Extract capacitance readings (63 sensors)
    cap_cols = [col for col in df.columns if col.startswith('cap_site_')]
    capacitances = df[cap_cols].values  # (T, 63)
    
    # Extract commanded velocity
    cmd_vel = df[['cmd_vx', 'cmd_vy', 'cmd_vz']].values  # (T, 3)
    
    # Also extract floating base state for reference (even though simulated)
    base_pos = df[['qpos_0', 'qpos_1', 'qpos_2']].values  # (T, 3)
    base_quat = df[['qpos_3', 'qpos_4', 'qpos_5', 'qpos_6']].values  # (T, 4)
    base_linvel = df[['qvel_0', 'qvel_1', 'qvel_2']].values  # (T, 3)
    base_angvel = df[['qvel_3', 'qvel_4', 'qvel_5']].values  # (T, 3)
    
    return {
        'time': time,
        'robot_qpos': robot_qpos,      # Now (T, 21)
        'robot_qvel': robot_qvel,      # Now (T, 21)
        'base_pos': base_pos,
        'base_quat': base_quat,
        'base_linvel': base_linvel,
        'base_angvel': base_angvel,
        'obstacle_pos': obstacle_pos,
        'obstacle_vel': obstacle_vel,
        'obstacle_radius': obstacle_radius,
        'capacitances': capacitances,
        'cmd_vel': cmd_vel,
    }
  
  def sample_trajectory(self, rng: jax.Array) -> Dict[str, jax.Array]:
    """Randomly sample one reference trajectory.
    
    Args:
      rng: JAX random key
    
    Returns:
      Dictionary with trajectory data as JAX arrays
    """
    traj_idx = jax.random.randint(rng, (), 0, self.num_trajectories)
    return {k: v[traj_idx] for k, v in self.stacked_trajectories.items()}
  
  def get_reference_at_step(
      self,
      trajectory: Dict[str, jax.Array],
      step: int
  ) -> Dict[str, jax.Array]:
    """Extract reference values at specific timestep.
    
    Args:
      trajectory: Trajectory dictionary with JAX arrays
      step: Current step index
    
    Returns:
      Dictionary with reference values at timestep (all scalar/vector, not trajectories)
    """
    # Get trajectory length
    traj_len = trajectory['time'].shape[0]
    
    # Clamp step to valid range [0, traj_len-1]
    step_clamped = jp.clip(step, 0, traj_len - 1)
    
    # Extract values at step
    ref = {
        'time': trajectory['time'][step_clamped],
        'robot_qpos': trajectory['robot_qpos'][step_clamped],  # (21,)
        'robot_qvel': trajectory['robot_qvel'][step_clamped],  # (21,)
        'base_pos': trajectory['base_pos'][step_clamped],      # (3,)
        'base_quat': trajectory['base_quat'][step_clamped],    # (4,)
        'base_linvel': trajectory['base_linvel'][step_clamped],  # (3,)
        'base_angvel': trajectory['base_angvel'][step_clamped],  # (3,)
        'obstacle_pos': trajectory['obstacle_pos'][step_clamped],  # (3,)
        'obstacle_vel': trajectory['obstacle_vel'][step_clamped],  # (3,)
        'obstacle_radius': trajectory['obstacle_radius'][step_clamped],  # scalar
        'capacitances': trajectory['capacitances'][step_clamped],  # (63,)
        'cmd_vel': trajectory['cmd_vel'][step_clamped],  # (3,)
    }
    
    return ref
  
  def get_initial_state(
      self,
      trajectory: Dict[str, jax.Array],
      start_step: int = 0
  ) -> Dict[str, jax.Array]:
    """Get initial state for reset from trajectory.
    
    Args:
      trajectory: Trajectory dictionary
      start_step: Which step to start from (default 0)
    
    Returns:
      Dictionary with initial state values
    """
    return self.get_reference_at_step(trajectory, start_step)
  
  def get_trajectory_length(self, trajectory: Dict[str, jax.Array]) -> int:
    """Get length of a trajectory.
    
    Args:
      trajectory: Trajectory dictionary
    
    Returns:
      Number of timesteps in trajectory
    """
    # return int(trajectory['time'].shape[0])
    return jp.sum(trajectory['time'] != trajectory['time'][-1])
  
  def get_stats(self) -> Dict[str, any]:
    """Get statistics about loaded trajectories.
    
    Returns:
      Dictionary with database statistics
    """
    return {
        'num_trajectories': self.num_trajectories,
        'min_length': int(self.trajectory_lengths.min()),
        'max_length': int(self.trajectory_lengths.max()),
        'mean_length': float(self.trajectory_lengths.mean()),
        'std_length': float(self.trajectory_lengths.std()),
        'total_timesteps': int(self.trajectory_lengths.sum()),
    }
  
  def validate_trajectory(self, trajectory: Dict[str, jax.Array]) -> bool:
    """Validate that a trajectory has expected shape and values.
    
    Args:
      trajectory: Trajectory dictionary
    
    Returns:
      True if valid, False otherwise (with warnings)
    """
    valid = True
    T = trajectory['time'].shape[0]
    
    # Check shapes
    expected_shapes = {
        'time': (T,),
        'robot_qpos': (T, 21),
        'robot_qvel': (T, 21),
        'base_pos': (T, 3),
        'base_quat': (T, 4),
        'base_linvel': (T, 3),
        'base_angvel': (T, 3),
        'obstacle_pos': (T, 3),
        'obstacle_vel': (T, 3),
        'obstacle_radius': (T,),
        'capacitances': (T, 63),
        'cmd_vel': (T, 3),
    }
    
    for key, expected_shape in expected_shapes.items():
      actual_shape = trajectory[key].shape
      if actual_shape != expected_shape:
        warnings.warn(f"Shape mismatch for {key}: expected {expected_shape}, got {actual_shape}")
        valid = False
    
    # Check for NaNs or Infs
    for key, arr in trajectory.items():
      if jp.any(jp.isnan(arr)) or jp.any(jp.isinf(arr)):
        warnings.warn(f"Found NaN or Inf in {key}")
        valid = False
    
    return valid


# Utility function for creating trajectory database with proper downsampling
def create_trajectory_database(
    traj_dir: str,
    control_freq_hz: float = 100.0,
    mpc_freq_hz: float = 500.0
) -> TrajectoryDatabase:
  """Create trajectory database with automatic downsampling.
  
  Args:
    traj_dir: Directory with trajectory CSV files
    control_freq_hz: Target control frequency (e.g., 50Hz for 0.02s timestep)
    mpc_freq_hz: MPC trajectory frequency (500Hz from your logs)
  
  Returns:
    TrajectoryDatabase instance
  """
  downsample_factor = int(mpc_freq_hz / control_freq_hz)
  print(f"Downsampling trajectories by {downsample_factor}x: {mpc_freq_hz}Hz -> {control_freq_hz}Hz")
  
  return TrajectoryDatabase(traj_dir, downsample_factor=downsample_factor)

# -----------------------------------------------------------
# Main Environment Class
# -----------------------------------------------------------

class Avoid(h12_skin_base.H12SkinEnv):
  """Collision avoidance environment with reference trajectory tracking."""
  
  def __init__(
      self,
      config: config_dict.ConfigDict = None,  # Will use default_config() from skeleton
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    """Initialize environment.
    
    Sets up the MuJoCo model, trajectory database, and observation/action spaces.
    """
    if config is None:
      config = default_config()
    # Update XML path to use h12_skin scene
    # TODO: Make sure this XML includes the obstacle and all 63 skin sensors
    super().__init__(
        xml_path=consts.FEET_ONLY_XML.as_posix(),  # Or your custom h12_skin XML
        config=config,
        config_overrides=config_overrides,
    )
    self._config = config
    self._post_init()
  
  def _post_init(self) -> None:
    """Post-initialization setup.
    Loads trajectory database, extracts model IDs, and sets up tracking.
    """
    # 1. Load trajectory database
    self._traj_db = create_trajectory_database(
        self._config.traj_dir,
        control_freq_hz=1.0 / self._config.ctrl_dt,
        mpc_freq_hz=500.0
    )
    print(f"Loaded trajectory database: {self._traj_db.get_stats()}")
    
    # 2. Get obstacle body/geom IDs
    self._obstacle_body_id = self._mj_model.body("obstacle").id
    obstacle_body = self._mj_model.body(self._obstacle_body_id)
    self._obstacle_geom_id = self._mj_model.geom("obstacle_geom").id
    
    # Get obstacle joint info for setting position/velocity
    self._obstacle_jnt_id = obstacle_body.jntadr[0]  # First joint of obstacle body
    self._obstacle_qpos_adr = self._mj_model.jnt_qposadr[self._obstacle_jnt_id]
    self._obstacle_qvel_adr = self._mj_model.jnt_dofadr[self._obstacle_jnt_id]
    
    # 3. Get skin sensor site IDs (63 sensors)
    skin_site_ids_np = get_skin_site_ids(self._mj_model)
    self._skin_site_ids = jp.array(skin_site_ids_np)
    self._num_skin_sensors = len(skin_site_ids_np)
    print(f"Found {self._num_skin_sensors} skin sensors")
    
    # 4. Define controlled joints (21 out of 27 total)
    # Exclude: wrists (24, 25, 26, 31, 32, 33)
    self._controlled_joint_indices = jp.array([
        7, 8, 9, 10, 11, 12,     # Left leg (6)
        13, 14, 15, 16, 17, 18,  # Right leg (6)
        19,                       # Torso (1)
        20, 21, 22, 23,               # Left arm: shoulder pitch/roll/yaw, elbow (3)
        27, 28, 29, 30,               # Right arm: shoulder pitch/roll/yaw, elbow (3)
    ])
    self._num_controlled_joints = 21
    
    # Corresponding qvel indices (qvel is offset by -1 from qpos after floating base)
    self._controlled_qvel_indices = self._controlled_joint_indices - 1
    
    # 5. Store default pose and control ranges (only for controlled joints)
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    full_default_pose = self._init_q[7:34]  # All 27 joints
    self._default_pose = full_default_pose[self._controlled_joint_indices - 7]  # Extract 19 joints
    
    # Control ranges for controlled joints only
    # Assuming actuators are in same order as joints
    self._lowers = jp.array(self._mj_model.actuator_ctrlrange[:, 0])[self._controlled_joint_indices - 7]
    self._uppers = jp.array(self._mj_model.actuator_ctrlrange[:, 1])[self._controlled_joint_indices - 7]
    
    # 6. Get torso body ID for COM tracking
    self._torso_body_id = self._mj_model.body("torso_link").id
    
    # 7. History sampling parameters (for 50Hz history from 500Hz control)
    # With ctrl_dt=0.02 (50Hz control), we want history every 10 steps
    self._history_delta = int(0.20 / self._config.ctrl_dt)  # Every 10 steps at 50Hz
    self._history_delta = 1

    # 8. Capacitance config
    self._cap_eps = self._config.capacitance_config.eps
    self._cap_sensing_radius = self._config.capacitance_config.sensing_radius
    self._cap_collision_threshold = self._config.capacitance_config.collision_threshold
    self._cap_safe_threshold = self._config.capacitance_config.safe_threshold
  
  def reset(self, rng: jax.Array) -> mjx_env.State:
    """Reset environment and sample new reference trajectory.
    Initializes robot and obstacle from MPC trajectory, sets up history buffers,
    and returns initial state.
    """
    rng, traj_rng, noise_rng = jax.random.split(rng, 3)
    
    # 1. Sample random reference trajectory
    current_trajectory = self._traj_db.sample_trajectory(traj_rng)
    traj_length = self._traj_db.get_trajectory_length(current_trajectory)
    
    # Get initial reference state (step 0)
    initial_ref = self._traj_db.get_initial_state(current_trajectory, start_step=0)
    
    # 2. Initialize robot state from trajectory
    # Start with home keyframe qpos
    qpos = self._init_q.copy()
    
    # Set robot joint positions from reference (only 21 controlled joints)
    # initial_ref['robot_qpos'] is (21,), need to place at controlled indices
    qpos = qpos.at[self._controlled_joint_indices].set(initial_ref['robot_qpos'])
    
    # Set obstacle position (qpos 34-40: x, y, z, qw, qx, qy, qz for freejoint)
    qpos = qpos.at[self._obstacle_qpos_adr:self._obstacle_qpos_adr+3].set(
        initial_ref['obstacle_pos']
    )
    # Set obstacle orientation to identity quaternion [1, 0, 0, 0]
    qpos = qpos.at[self._obstacle_qpos_adr+3:self._obstacle_qpos_adr+7].set(
        jp.array([1.0, 0.0, 0.0, 0.0])
    )
    
    # Initialize velocities
    qvel = jp.zeros(self.mjx_model.nv)
    
    # Set robot joint velocities from reference (only 21 controlled joints)
    qvel = qvel.at[self._controlled_qvel_indices].set(initial_ref['robot_qvel'])
    
    # Set obstacle velocity (qvel for freejoint: 3 linear + 3 angular)
    qvel = qvel.at[self._obstacle_qvel_adr:self._obstacle_qvel_adr+3].set(
        initial_ref['obstacle_vel']
    )
    
    # 3. Create MJX data and step forward
    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        impl=self.mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)
    
    # 4. Compute initial capacitances
    capacitances = compute_all_capacitances(
        data,
        self._skin_site_ids,
        self._obstacle_body_id,
        self._obstacle_geom_id,
        self.mjx_model,
        eps=self._cap_eps,
        sensing_radius=self._cap_sensing_radius
    )
    
    # 5. Initialize history buffers (now 21 joints instead of 27)
    history_len = self._config.history_len
    qpos_error_history = jp.zeros(history_len * self._num_controlled_joints)
    qvel_history = jp.zeros(history_len * self._num_controlled_joints)
    action_history = jp.zeros(history_len * self._num_controlled_joints)
    capacitance_history = jp.zeros(history_len * self._num_skin_sensors)
    
    # 6. Set up info dict
    info = {
        'rng': rng,
        'current_trajectory': current_trajectory,
        'traj_step': 0,
        'traj_length': traj_length,
        'last_act': jp.zeros(self._num_controlled_joints),
        'last_last_act': jp.zeros(self._num_controlled_joints),
        'motor_targets': jp.zeros(self._num_controlled_joints),
        'qpos_error_history': qpos_error_history,
        'qvel_history': qvel_history,
        'action_history': action_history,
        'capacitance_history': capacitance_history,
        'capacitances': capacitances,
        'history_counter': 0,  # Tracks when to sample history
    }
    
    # 7. Initialize metrics
    metrics = {}
    reward_keys = self._config.reward_config.scales.keys()
    for key in reward_keys:
        metrics[f'reward/{key}'] = jp.zeros(())
    
    # Additional metrics
    metrics['collision_detected'] = jp.zeros(())
    metrics['min_capacitance'] = jp.zeros(())
    metrics['tracking_error'] = jp.zeros(())
    
    # 8. Get initial observation
    obs = self._get_obs(data, info, noise_rng)
    
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)
  
  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    """Step environment forward.
    Applies action, steps physics, computes rewards, and updates state.
    """
    rng, noise_rng = jax.random.split(state.info['rng'])
    
    # 1. Apply action (offset from default pose, with scaling and clipping)
    motor_targets = self._default_pose + action * self._config.action_scale
    motor_targets = jp.clip(motor_targets, self._lowers, self._uppers)
    
    # Create full control vector for all actuators
    # We need to set only the controlled joints, others stay at default
    full_ctrl = self._init_q[7:34].copy()  # Start with all 27 joints at default
    # Map 21 motor targets to their positions in the full 27-joint array
    ctrl_indices_in_full = self._controlled_joint_indices - 7  # Convert to 0-based array indices
    full_ctrl = full_ctrl.at[ctrl_indices_in_full].set(motor_targets)
    
    # 2. Step physics for n_substeps
    data = mjx_env.step(
        self.mjx_model, state.data, full_ctrl, self.n_substeps
    )
    state.info['motor_targets'] = motor_targets
    
    # 3. Increment trajectory step counter
    traj_step = state.info['traj_step'] + 1
    state.info['traj_step'] = traj_step
    
    # 4. Compute capacitances for current state
    capacitances = compute_all_capacitances(
        data,
        self._skin_site_ids,
        self._obstacle_body_id,
        self._obstacle_geom_id,
        self.mjx_model,
        eps=self._cap_eps,
        sensing_radius=self._cap_sensing_radius
    )
    state.info['capacitances'] = capacitances
    
    # 5. Get reference values for current step
    ref_state = self._traj_db.get_reference_at_step(
        state.info['current_trajectory'],
        traj_step
    )
    
    # 6. Update history buffers (sample every history_delta steps)
    # Extract only controlled joints from current state
    current_qvel = data.qvel[self._controlled_qvel_indices]
    current_qpos = data.qpos[self._controlled_joint_indices]
    
    history_counter = state.info['history_counter'] + 1
    should_sample = (history_counter % self._history_delta) == 0
    
    # Roll and update histories (now using 21 joints)
    qvel_history = jp.where(
        should_sample,
        jp.roll(state.info['qvel_history'], self._num_controlled_joints).at[:self._num_controlled_joints].set(current_qvel),
        state.info['qvel_history']
    )
    
    qpos_error = current_qpos - state.info['motor_targets']
    qpos_error_history = jp.where(
        should_sample,
        jp.roll(state.info['qpos_error_history'], self._num_controlled_joints).at[:self._num_controlled_joints].set(qpos_error),
        state.info['qpos_error_history']
    )
    
    action_history = jp.where(
        should_sample,
        jp.roll(state.info['action_history'], self._num_controlled_joints).at[:self._num_controlled_joints].set(action),
        state.info['action_history']
    )
    
    capacitance_history = jp.where(
        should_sample,
        jp.roll(state.info['capacitance_history'], self._num_skin_sensors).at[:self._num_skin_sensors].set(capacitances),
        state.info['capacitance_history']
    )
    
    state.info['qvel_history'] = qvel_history
    state.info['qpos_error_history'] = qpos_error_history
    state.info['action_history'] = action_history
    state.info['capacitance_history'] = capacitance_history
    state.info['history_counter'] = history_counter
    
    # 7. Compute observations
    obs = self._get_obs(data, state.info, noise_rng)
    
    # 8. Compute rewards
    rewards = self._get_reward(data, action, state.info, ref_state)
    
    # Scale rewards (Go1 style)
    rewards = {
        k: v * self._config.reward_config.scales[k] 
        for k, v in rewards.items()
    }
    
    # Combine ALL rewards (not split pos/neg)
    reward = jp.clip(sum(rewards.values()) * self.dt, -10000.0, 10000.0)    
    
    # 9. Check termination
    done = self._get_termination(data, capacitances)
    
    # 10. Update metrics
    all_rewards = rewards
    for k, v in all_rewards.items():
        state.metrics[f'reward/{k}'] = v
    
    # Additional metrics
    state.metrics['collision_detected'] = jp.any(capacitances > self._cap_collision_threshold).astype(jp.float32)
    state.metrics['min_capacitance'] = jp.min(jp.where(capacitances > 0, capacitances, jp.inf))
    state.metrics['tracking_error'] = jp.linalg.norm(current_qpos - ref_state['robot_qpos'])
    
    # 11. Update state info
    state.info['last_last_act'] = state.info['last_act']
    state.info['last_act'] = action
    state.info['rng'] = rng
    
    # 12. Return updated state
    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state
  
  # Property overrides
  @property
  def action_size(self) -> int:
    """Action space size: 21 actuated joints."""
    return 21
  
  def _get_obs(
      self,
      data,  # mjx.Data
      info: Dict[str, Any],
      rng: jax.Array,
  ) -> Dict[str, jax.Array]:
    """Construct observations for both actor and critic.
    
    Returns dict with 'state' (actor obs) and 'privileged_state' (critic obs).
    """
    # Get current reference state
    ref_state = self._traj_db.get_reference_at_step(
        info['current_trajectory'],
        info['traj_step']
    )
    
    # ============ ACTOR OBSERVATION ============
    # Extract only controlled joints
    joint_pos = data.qpos[self._controlled_joint_indices]  # (21,)
    joint_vel = data.qvel[self._controlled_qvel_indices]  # (21,)
    
    # Base sensor readings (already in body frame)
    gyro = self.get_gyro(data)  # (3,)
    gravity = self.get_gravity(data)  # (3,)
    
    # Joint offsets from default
    joint_pos_offset = joint_pos - self._default_pose  # (21,)
    
    # Capacitance readings
    capacitances = info['capacitances']  # (63,)
    
    # History buffers (already stored in info)
    qpos_error_history = info['qpos_error_history']  # (history_len * 21,)
    qvel_history = info['qvel_history']  # (history_len * 21,)
    action_history = info['action_history']  # (history_len * 21,)
    capacitance_history = info['capacitance_history']  # (history_len * 63,)
    
    # Combine base proprioception (before noise)
    base_proprio = jp.concatenate([
        gyro,                    # 3
        gravity,                 # 3
        joint_pos_offset,        # 21
        joint_vel,               # 21
        qpos_error_history,      # history_len * 21
        qvel_history,            # history_len * 21
        action_history,          # history_len * 21
        capacitances,            # 63
        capacitance_history,     # history_len * 63
    ])
    
    # ============ Add Observation Noise ============
    noise_level = self._config.obs_noise.level
    noise_scales = self._config.obs_noise.scales
    
    # Build noise vector
    noise_vec = jp.zeros_like(base_proprio)
    idx = 0
    
    # Gyro noise
    noise_vec = noise_vec.at[idx:idx+3].set(noise_level * noise_scales.gyro)
    idx += 3
    
    # Gravity noise
    noise_vec = noise_vec.at[idx:idx+3].set(noise_level * noise_scales.gravity)
    idx += 3
    
    # Joint position noise (21 joints)
    noise_vec = noise_vec.at[idx:idx+self._num_controlled_joints].set(noise_level * noise_scales.joint_pos)
    idx += self._num_controlled_joints
    
    # Joint velocity noise (21 joints)
    noise_vec = noise_vec.at[idx:idx+self._num_controlled_joints].set(noise_level * noise_scales.joint_vel)
    idx += self._num_controlled_joints
    
    # History buffers - use same noise as their base quantities
    history_len = self._config.history_len
    
    # qpos_error_history
    for i in range(history_len):
      noise_vec = noise_vec.at[idx+i*self._num_controlled_joints:idx+(i+1)*self._num_controlled_joints].set(
          noise_level * noise_scales.joint_pos
      )
    idx += history_len * self._num_controlled_joints
    
    # qvel_history
    for i in range(history_len):
      noise_vec = noise_vec.at[idx+i*self._num_controlled_joints:idx+(i+1)*self._num_controlled_joints].set(
          noise_level * noise_scales.joint_vel
      )
    idx += history_len * self._num_controlled_joints
    
    # action_history - no noise on past actions
    idx += history_len * self._num_controlled_joints
    
    # Capacitance noise
    noise_vec = noise_vec.at[idx:idx+63].set(noise_level * noise_scales.capacitance)
    idx += 63
    
    # Capacitance history
    for i in range(history_len):
      noise_vec = noise_vec.at[idx+i*63:idx+(i+1)*63].set(noise_level * noise_scales.capacitance)
    idx += history_len * 63
    
    # Apply uniform noise [-1, 1] * noise_vec
    rng, noise_rng = jax.random.split(rng)
    noise = (2 * jax.random.uniform(noise_rng, shape=base_proprio.shape) - 1) * noise_vec
    noisy_proprio = base_proprio + noise
    
    # ============ Partial Reference Information ============
    partial_ref = jp.concatenate([
        ref_state['robot_qvel'],      # 21
        ref_state['base_angvel'],     # 3
        ref_state['robot_qpos'],      # 21
    ])
    
    # Concatenate actor observation
    actor_obs = jp.concatenate([noisy_proprio, partial_ref])
    
    # ============ CRITIC OBSERVATION (Privileged) ============
    
    # True base state (from floating base qpos/qvel)
    base_pos = data.qpos[0:3]  # (3,)
    base_quat = data.qpos[3:7]  # (4,)
    base_linvel = data.qvel[0:3]  # (3,)
    base_angvel = data.qvel[3:6]  # (3,)
    
    # Gravity vector
    gravity = self.get_gravity(data)  # (3,)
    
    # Joint state (current, 21 joints)
    joint_vel = data.qvel[self._controlled_qvel_indices]  # (21,)
    
    # History buffers (no noise)
    joint_pos_history = info['qpos_error_history']  # (history_len * 21,)
    qvel_history = info['qvel_history']  # (history_len * 21,)
    action_history = info['action_history']  # (history_len * 21,)
    
    # True capacitance readings (no noise)
    capacitances = info['capacitances']  # (63,)
    capacitance_history = info['capacitance_history']  # (history_len * 63,)
    
    # True obstacle state
    obstacle_pos = data.qpos[self._obstacle_qpos_adr:self._obstacle_qpos_adr+3]  # (3,)
    obstacle_vel = data.qvel[self._obstacle_qvel_adr:self._obstacle_qvel_adr+3]  # (3,)
    obstacle_radius = self.mjx_model.geom_size[self._obstacle_geom_id, 0]  # scalar
    
    # Combine privileged proprioception
    priv_proprio = jp.concatenate([
        base_pos,                # 3
        base_quat,               # 4
        base_linvel,             # 3
        base_angvel,             # 3
        gravity,                 # 3
        joint_pos_history,       # history_len * 21
        joint_vel,               # 21
        action_history,          # history_len * 21
        capacitances,            # 63
        capacitance_history,     # history_len * 63
        obstacle_pos,            # 3
        obstacle_vel,            # 3
        jp.array([obstacle_radius]),  # 1
    ])
    
    # ============ Full Reference State ============
    full_ref = jp.concatenate([
        ref_state['base_pos'],        # 3
        ref_state['base_quat'],       # 4
        ref_state['base_linvel'],     # 3
        ref_state['base_angvel'],     # 3
        ref_state['robot_qpos'],      # 21
        ref_state['robot_qvel'],      # 21
        ref_state['capacitances'],    # 63
    ])
    
    # Concatenate critic observation
    critic_obs = jp.concatenate([priv_proprio, full_ref])
    
    return {
        "state": actor_obs,
        "privileged_state": critic_obs,
    }


  def _compute_obs_size(self) -> tuple:
    """Compute observation sizes for actor and critic.
    
    Returns:
      (actor_obs_size, critic_obs_size)
    """
    history_len = self._config.history_len
    num_joints = self._num_controlled_joints  # 21
    
    # Actor observation size
    actor_proprio_size = (
        3 +  # gyro
        3 +  # gravity
        num_joints +  # joint_pos_offset
        num_joints +  # joint_vel
        history_len * num_joints +  # qpos_error_history
        history_len * num_joints +  # qvel_history
        history_len * num_joints +  # action_history
        63 +  # capacitances
        history_len * 63  # capacitance_history
    )
    
    actor_ref_size = (
        num_joints +  # ref_joint_vel
        3 +   # ref_base_angvel
        num_joints    # ref_joint_pos
    )
    
    actor_obs_size = actor_proprio_size + actor_ref_size
    
    # Critic observation size
    critic_proprio_size = (
        3 +  # base_pos
        4 +  # base_quat
        3 +  # base_linvel
        3 +  # base_angvel
        3 +  # gravity
        history_len * num_joints +  # joint_pos_history
        num_joints +  # joint_vel
        history_len * num_joints +  # action_history
        63 +  # capacitances
        history_len * 63 +  # capacitance_history
        3 +  # obstacle_pos
        3 +  # obstacle_vel
        1    # obstacle_radius
    )
    
    critic_ref_size = (
        3 +  # ref_base_pos
        4 +  # ref_base_quat
        3 +  # ref_base_linvel
        3 +  # ref_base_angvel
        num_joints +  # ref_joint_pos
        num_joints +  # ref_joint_vel
        63    # ref_capacitances
    )
    
    critic_obs_size = critic_proprio_size + critic_ref_size
    
    return actor_obs_size, critic_obs_size

  # Optional: Helper function for visualization/debugging
  def _format_observation(self, obs: jax.Array, is_critic: bool = False) -> Dict[str, Any]:
    """Parse observation vector into named components for debugging.
    
    Args:
      obs: Observation vector (actor or critic)
      is_critic: Whether this is a critic observation
    
    Returns:
      Dictionary with named observation components
    """
    history_len = self._config.history_len
    idx = 0
    
    if not is_critic:
      # Actor observation
      components = {}
      components['gyro'] = obs[idx:idx+3]
      idx += 3
      components['gravity'] = obs[idx:idx+3]
      idx += 3
      components['joint_pos_offset'] = obs[idx:idx+21]
      idx += 21
      components['joint_vel'] = obs[idx:idx+21]
      idx += 21
      components['qpos_error_history'] = obs[idx:idx+history_len*21]
      idx += history_len * 21
      components['qvel_history'] = obs[idx:idx+history_len*21]
      idx += history_len * 21
      components['action_history'] = obs[idx:idx+history_len*21]
      idx += history_len * 21
      components['capacitances'] = obs[idx:idx+63]
      idx += 63
      components['capacitance_history'] = obs[idx:idx+history_len*63]
      idx += history_len * 63
      components['ref_joint_vel'] = obs[idx:idx+21]
      idx += 21
      components['ref_base_angvel'] = obs[idx:idx+3]
      idx += 3
      components['ref_joint_pos'] = obs[idx:idx+21]
      idx += 21
    else:
      # Critic observation
      components = {}
      components['base_pos'] = obs[idx:idx+3]
      idx += 3
      components['base_quat'] = obs[idx:idx+4]
      idx += 4
      components['base_linvel'] = obs[idx:idx+3]
      idx += 3
      components['base_angvel'] = obs[idx:idx+3]
      idx += 3
      components['gravity'] = obs[idx:idx+3]
      idx += 3
      components['joint_pos_history'] = obs[idx:idx+history_len*21]
      idx += history_len * 21
      components['joint_vel'] = obs[idx:idx+21]
      idx += 21
      components['action_history'] = obs[idx:idx+history_len*21]
      idx += history_len * 21
      components['capacitances'] = obs[idx:idx+63]
      idx += 63
      components['capacitance_history'] = obs[idx:idx+history_len*63]
      idx += history_len * 63
      components['obstacle_pos'] = obs[idx:idx+3]
      idx += 3
      components['obstacle_vel'] = obs[idx:idx+3]
      idx += 3
      components['obstacle_radius'] = obs[idx:idx+1]
      idx += 1
      components['ref_base_pos'] = obs[idx:idx+3]
      idx += 3
      components['ref_base_quat'] = obs[idx:idx+4]
      idx += 4
      components['ref_base_linvel'] = obs[idx:idx+3]
      idx += 3
      components['ref_base_angvel'] = obs[idx:idx+3]
      idx += 3
      components['ref_joint_pos'] = obs[idx:idx+21]
      idx += 21
      components['ref_joint_vel'] = obs[idx:idx+21]
      idx += 21
      components['ref_capacitances'] = obs[idx:idx+63]
      idx += 63
    
    return components
    
  def _get_reward(
      self,
      data,  # mjx.Data
      action: jax.Array,
      info: Dict[str, Any],
      ref_state: Dict[str, jax.Array],
  ) -> Dict[str, jax.Array]:  # Returns single dict, not tuple
    """Compute all reward components.
    
    Handles two phases:
    1. Tracking phase: While MPC trajectory is available, track reference
    2. Recovery phase: After trajectory ends, recover to stable standing
    
    Args:
      data: Current MJX state
      action: Current action (21,)
      info: State info dict
      ref_state: Reference state from trajectory at current timestep
    
    Returns:
      Dictionary with all reward components
    """
    # Get capacitances
    capacitances = info['capacitances']
    
    # Extract controlled joints from current state
    current_qpos = data.qpos[self._controlled_joint_indices]  # (21,)
    current_qvel = data.qvel[self._controlled_qvel_indices]  # (21,)
    
    # ============ PHASE DETECTION ============
    traj_step = info['traj_step']
    traj_length = info['traj_length']
    in_tracking_phase = traj_step < traj_length
    in_recovery_phase = ~in_tracking_phase
    
    # ============ TRACKING REWARDS (masked by phase) ============
    tracking_rewards = {
        'joint_pos_tracking': jp.where(
            in_tracking_phase,
            self._reward_joint_pos_tracking(current_qpos, ref_state['robot_qpos']),
            0.0
        ),
        'joint_vel_tracking': jp.where(
            in_tracking_phase,
            self._reward_joint_vel_tracking(current_qvel, ref_state['robot_qvel']),
            0.0
        ),
        'base_vel_tracking': jp.where(
            in_tracking_phase,
            self._reward_base_vel_tracking(data.qvel[0:3], ref_state['base_linvel']),
            0.0
        ),
        'base_angvel_tracking': jp.where(
            in_tracking_phase,
            self._reward_base_angvel_tracking(data.qvel[3:6], ref_state['base_angvel']),
            0.0
        ),
        'torque_tracking': jp.where(
            in_tracking_phase,
            self._reward_torque_tracking(data, ref_state),
            0.0
        ),
    }
    
    # ============ RECOVERY REWARDS (masked by phase) ============
    recovery_rewards = {
        'recovery_upright': jp.where(
            in_recovery_phase,
            self._reward_upright(data),
            0.0
        ),
        'recovery_height': jp.where(
            in_recovery_phase,
            self._reward_height(data),
            0.0
        ),
        'recovery_posture': jp.where(
            in_recovery_phase,
            self._reward_posture(current_qpos),
            0.0
        ),
        'recovery_stability': jp.where(
            in_recovery_phase,
            self._reward_stability(data),
            0.0
        ),
        'recovery_feet_contact': jp.where(
            in_recovery_phase,
            self._reward_feet_contact(data),
            0.0
        ),
    }
    
    # ============ ALWAYS-ACTIVE REWARDS & COSTS ============
    always_active = {
        # Avoidance rewards
        'clearance_reward': self._reward_clearance(capacitances),
        
        # Collision and proximity penalties
        'collision_penalty': self._cost_collision(capacitances),
        'proximity_penalty': self._cost_proximity(capacitances),
        
        # Regularization costs
        'action_rate': self._cost_action_rate(action, info['last_act']),
        'torque': self._cost_torque(data),
        'energy': self._cost_energy(data),
        'joint_limit': self._cost_joint_limit(current_qpos),
    }
    
    # ============ COMBINE ALL ============
    all_rewards = {**tracking_rewards, **recovery_rewards, **always_active}
    
    return all_rewards

  # ============================================================
  # TRACKING REWARDS (Positive, Exponential)
  # ============================================================

  def _reward_joint_pos_tracking(
      self,
      qpos: jax.Array,
      ref_qpos: jax.Array,
      sigma: float = 0.1
  ) -> jax.Array:
    """Exponential reward for joint position tracking.
    
    Mirrors MPC residual: minimize ||qpos - ref_qpos||^2
    
    Args:
      qpos: Current joint positions (21,)
      ref_qpos: Reference joint positions (21,)
      sigma: Temperature parameter
    
    Returns:
      Reward in [0, 1], higher is better
    """
    error = qpos - ref_qpos
    squared_error = jp.sum(error ** 2)
    return jp.exp(-squared_error / sigma)


  def _reward_joint_vel_tracking(
      self,
      qvel: jax.Array,
      ref_qvel: jax.Array,
      sigma: float = 0.5
  ) -> jax.Array:
    """Exponential reward for joint velocity tracking.
    
    Args:
      qvel: Current joint velocities (21,)
      ref_qvel: Reference joint velocities (21,)
      sigma: Temperature parameter
    
    Returns:
      Reward in [0, 1]
    """
    error = qvel - ref_qvel
    squared_error = jp.sum(error ** 2)
    return jp.exp(-squared_error / sigma)


  def _reward_base_vel_tracking(
      self,
      base_linvel: jax.Array,
      ref_base_linvel: jax.Array,
      sigma: float = 0.25
  ) -> jax.Array:
    """Exponential reward for base linear velocity tracking.
    
    Args:
      base_linvel: Current base linear velocity (3,)
      ref_base_linvel: Reference base linear velocity (3,)
      sigma: Temperature parameter
    
    Returns:
      Reward in [0, 1]
    """
    error = base_linvel - ref_base_linvel
    squared_error = jp.sum(error ** 2)
    return jp.exp(-squared_error / sigma)


  def _reward_base_angvel_tracking(
      self,
      base_angvel: jax.Array,
      ref_base_angvel: jax.Array,
      sigma: float = 0.25
  ) -> jax.Array:
    """Exponential reward for base angular velocity tracking.
    
    Args:
      base_angvel: Current base angular velocity (3,)
      ref_base_angvel: Reference base angular velocity (3,)
      sigma: Temperature parameter
    
    Returns:
      Reward in [0, 1]
    """
    error = base_angvel - ref_base_angvel
    squared_error = jp.sum(error ** 2)
    return jp.exp(-squared_error / sigma)


  def _reward_torque_tracking(
      self,
      data,  # mjx.Data
      ref_state: Dict[str, jax.Array],
      sigma: float = 1.0
  ) -> jax.Array:
    """Exponential reward for torque tracking (if reference torques available).
    
    Note: MPC trajectories may not include torques. If not available,
    this can be disabled or we can estimate from inverse dynamics.
    
    Args:
      data: Current MJX state
      ref_state: Reference state (may contain 'joint_torques' if logged)
      sigma: Temperature parameter
    
    Returns:
      Reward in [0, 1], or 0 if torques not available
    """
    # Check if reference torques are available
    if 'joint_torques' not in ref_state:
        return jp.array(0.0)
    
    # Get current actuator forces (torques) for controlled joints only
    current_torques = data.qfrc_actuator[self._controlled_qvel_indices]  # (21,) actuated joints
    ref_torques = ref_state['joint_torques']
    
    error = current_torques - ref_torques
    squared_error = jp.sum(error ** 2)
    return jp.exp(-squared_error / sigma)


  def _reward_clearance(
      self,
      capacitances: jax.Array,
      safe_threshold: float = None
  ) -> jax.Array:
    """Reward for maintaining safe clearance from obstacle.
    
    Rewards sensors that detect obstacle but maintain safe distance.
    
    Args:
      capacitances: (63,) capacitance readings
      safe_threshold: Capacitance threshold for "safe" distance
    
    Returns:
      Reward proportional to number of sensors in safe range
    """
    # Count sensors detecting obstacle (cap > 0) but below safe threshold
    in_range = capacitances > 0.0
    if safe_threshold is None:
      safe_threshold = self._cap_safe_threshold
    is_safe = capacitances < safe_threshold
    safe_detections = in_range & is_safe
    
    # Reward proportional to safe detections
    num_safe = jp.sum(safe_detections.astype(jp.float32))
    total_sensors = jp.float32(capacitances.shape[0])
    
    return num_safe / total_sensors


  # ============================================================
  # COLLISION & PROXIMITY PENALTIES (Negative Costs)
  # ============================================================

  def _cost_collision(
      self,
      capacitances: jax.Array,
      collision_threshold: float = None
  ) -> jax.Array:
    """Large penalty for collision with obstacle.
    
    Mirrors MPC collision avoidance constraint.
    
    Args:
      capacitances: (63,) capacitance readings
      collision_threshold: Threshold indicating collision (default from config)
    
    Returns:
      Large negative cost if collision detected, 0 otherwise
    """
    if collision_threshold is None:
      collision_threshold = self._cap_collision_threshold
    
    # Check if any sensor exceeds collision threshold
    collision_detected = jp.any(capacitances > collision_threshold)
    
    # Return 1.0 if collision (* -100.0 collision penalty), 0.0 otherwise
    return jp.where(collision_detected, 1.0, 0.0)


  def _cost_proximity(
      self,
      capacitances: jax.Array
  ) -> jax.Array:
    """Penalty proportional to proximity to obstacle.
    
    Mirrors MPC "Obstacle Proximity" residual.
    Uses squared capacitance (inverse distance squared) for stronger penalty when close.
    
    Args:
      capacitances: (63,) capacitance readings
    
    Returns:
      Negative cost proportional to sum of squared capacitances
    """
    # Filter positive capacitances (in sensing range)
    in_range = capacitances > 0.0
    
    # Compute sum of squared capacitances (stronger penalty when close)
    # proximity = jp.where(in_range, capacitances ** 2, 0.0)
    proximity = jp.where(in_range, capacitances, 0.0)
    total_proximity = jp.sum(proximity)
    
    # Normalize by number of sensors for consistent scale
    avg_proximity = total_proximity / jp.float32(capacitances.shape[0])
    
    # return -avg_proximity
    return avg_proximity


  # ============================================================
  # REGULARIZATION COSTS (Negative, for smooth/efficient motion)
  # ============================================================

  def _cost_action_rate(
      self,
      action: jax.Array,
      last_action: jax.Array
  ) -> jax.Array:
    """Penalize rapid action changes for smooth control.
    
    Args:
      action: Current action (21,)
      last_action: Previous action (21,)
    
    Returns:
      Negative cost proportional to action change
    """
    delta = action - last_action
    # return -jp.sum(delta ** 2)
    return jp.sum(delta ** 2)


  def _cost_torque(
      self,
      data  # mjx.Data
  ) -> jax.Array:
      """Penalize large torques for energy efficiency.
      
      Args:
        data: MJX state
      
      Returns:
        Negative cost proportional to squared torques
      """
      # Get actuator forces (torques) for controlled joints only
      torques = data.qfrc_actuator[self._controlled_qvel_indices]  # (21,) actuated joints
      # return -jp.sum(torques ** 2)
      return jp.sum(torques ** 2)


  def _cost_energy(
      self,
      data  # mjx.Data
  ) -> jax.Array:
      """Penalize energy consumption (power = torque * velocity).
      
      Args:
        data: MJX state
      
      Returns:
        Negative cost proportional to total power
      """
      torques = data.qfrc_actuator[self._controlled_qvel_indices]  # (21,)
      qvel = data.qvel[self._controlled_qvel_indices]  # (21,)
      power = torques * qvel
      # return -jp.sum(jp.abs(power))
      return jp.sum(jp.abs(power))

  def _cost_joint_limit(
      self,
      qpos: jax.Array,
      margin: float = 0.1
  ) -> jax.Array:
    """Penalize approaching joint limits.
    
    Args:
      qpos: Joint positions (21,)
      margin: Fraction of range to start penalizing (0.1 = 10% margin)
    
    Returns:
      Negative cost if approaching limits
    """
    # Compute normalized position in range [0, 1]
    normalized = (qpos - self._lowers) / (self._uppers - self._lowers)
    
    # Penalize if within margin of limits
    lower_violation = jp.maximum(0.0, margin - normalized)
    upper_violation = jp.maximum(0.0, normalized - (1.0 - margin))
    
    total_violation = jp.sum(lower_violation ** 2) + jp.sum(upper_violation ** 2)
    
    return -total_violation


  # ============================================================
  # TERMINATION CONDITION
  # ============================================================

  def _get_termination(
      self,
      data,  # mjx.Data
      capacitances: jax.Array
  ) -> jax.Array:
    """Check termination conditions.
    
    Terminate if:
    1. Joint limits exceeded
    2. Robot falls (low gravity_z)
    3. Robot moves too far from origin (optional safety)
    
    Args:
      data: MJX state
      capacitances: (63,) capacitance readings
    
    Returns:
      Boolean (as float): 1.0 if should terminate, 0.0 otherwise
    """
    # 1. Check joint limits for controlled joints only
    joint_angles = data.qpos[self._controlled_joint_indices]  # (21,)
    joint_limit_exceed = jp.any(joint_angles < self._lowers)
    joint_limit_exceed |= jp.any(joint_angles > self._uppers)
    
    # 2. Check if robot falls (gravity sensor z < threshold)
    gravity = self.get_gravity(data)
    fall_termination = gravity[2] < 0.85  # Torso tilted significantly
        
    # 4. Check if robot moved too far (optional, safety boundary)
    base_pos = data.qpos[0:3]
    distance_from_origin = jp.linalg.norm(base_pos[:2])  # xy distance
    too_far = distance_from_origin > 5.0  # 5m safety radius
    
    # Combine conditions
    terminate = joint_limit_exceed | fall_termination | too_far
    
    # Respect early_termination config flag
    terminate = jp.where(
        self._config.early_termination,
        terminate,
        joint_limit_exceed  # Always terminate on joint limits
    )
    
    return terminate.astype(jp.float32)

    # ============================================================
    # RECOVERY PHASE REWARDS (After trajectory ends)
    # ============================================================

  def _reward_upright(
        self,
        data,  # mjx.Data
    ) -> jax.Array:
    """Reward for maintaining upright orientation during recovery.
    
    Similar to Go1 getup task - robot should orient torso upright.
    
    Args:
        data: MJX state
    
    Returns:
        Reward in [0, 1] for being upright
    """
    gravity = self.get_gravity(data)  # (3,)
    
    # Want gravity to point down in body frame: [0, 0, 1]
    # (In world frame, torso z-axis points up)
    target = jp.array([0.0, 0.0, 1.0])
    
    error = jp.sum((gravity - target) ** 2)
    return jp.exp(-2.0 * error)


  def _reward_height(
        self,
        data,  # mjx.Data
        target_height: float = 1.03
    ) -> jax.Array:
    """Reward for achieving target torso height during recovery.
    
    Similar to Go1 getup task - robot should stand at normal height.
    
    Args:
        data: MJX state
        target_height: Desired torso height (meters)
    
    Returns:
        Reward for being at target height
    """
    torso_pos = data.xpos[self._torso_body_id]
    torso_height = torso_pos[2]
    
    # Clamp height to avoid rewarding being too high
    clamped_height = jp.minimum(torso_height, target_height)
    
    # Exponential reward for height
    return jp.exp(clamped_height) - 1.0


  def _reward_posture(
        self,
        joint_pos: jax.Array
    ) -> jax.Array:
    """Reward for returning to neutral standing pose during recovery.
    
    Args:
        joint_pos: Current joint positions (21,)
    
    Returns:
        Reward for matching default pose
    """
    error = jp.sum((joint_pos - self._default_pose) ** 2)
    return jp.exp(-0.5 * error)


  def _reward_stability(
      self,
      data,  # mjx.Data
  ) -> jax.Array:
    """Reward for stability during recovery (low velocities).
    
    Encourages robot to settle into a stable standing position.
    
    Args:
        data: MJX state
    
    Returns:
        Reward for low velocities
    """
    # Base linear velocity (xy only, ignore z)
    base_linvel_xy = data.qvel[0:2]
    
    # Base angular velocity
    base_angvel = data.qvel[3:6]
    
    # Joint velocities (only controlled joints)
    joint_vel = data.qvel[self._controlled_qvel_indices]  # (21,)
    
    # Penalize velocities
    lin_vel_cost = jp.sum(base_linvel_xy ** 2)
    ang_vel_cost = jp.sum(base_angvel ** 2)
    joint_vel_cost = jp.sum(joint_vel ** 2) * 0.1  # Lower weight on joint velocities
    
    total_vel_cost = lin_vel_cost + ang_vel_cost + joint_vel_cost
    
    return jp.exp(-total_vel_cost)

  def _reward_feet_contact(
        self,
        data,  # mjx.Data
    ) -> jax.Array:
    """Reward for having feet in contact with ground during recovery.
    
    Encourages stable standing with both feet on ground.
    
    Args:
        data: MJX state
    
    Returns:
        Reward for feet contact
    """
    # Get contact forces on feet (if you have contact sensors)
    # This is a simplified version - adjust based on your sensors
    
    # For now, check if feet are near ground level
    left_foot_z = data.site_xpos[self._mj_model.site("left_foot").id][2]
    right_foot_z = data.site_xpos[self._mj_model.site("right_foot").id][2]
    
    # Reward if feet are close to ground (< 5cm)
    left_contact = jp.exp(-50.0 * left_foot_z ** 2)
    right_contact = jp.exp(-50.0 * right_foot_z ** 2)
    
    return (left_contact + right_contact) / 2.0


  # ============================================================
  # OPTIONAL: Additional reward shaping functions
  # ============================================================

  def _reward_com_away_from_obstacle(
      self,
      data,  # mjx.Data
      capacitances: jax.Array,
      desired_offset: float = 0.2
  ) -> jax.Array:
    """Reward for shifting COM away from obstacle centroid.
    
    Mirrors MPC "CoM Away From Obstacle" residual.
    
    Args:
      data: MJX state
      capacitances: (63,) capacitance readings
      desired_offset: Desired distance in xy plane (meters)
    
    Returns:
      Reward for maintaining desired offset
    """    
    # Compute obstacle centroid from capacitances
    centroid, total_weight, num_detections = compute_obstacle_centroid(
        data, self._skin_site_ids, capacitances
    )
    
    # If no obstacle detected, return 0
    no_detection = total_weight <= 0.0
    
    # Get COM position
    com_pos = data.subtree_com[self._torso_body_id]  # (3,)
    
    # Compute direction from obstacle to COM (xy only)
    away_dir = com_pos[:2] - centroid[:2]
    current_offset = jp.linalg.norm(away_dir)
    
    # Reward for being at desired offset
    offset_error = jp.abs(current_offset - desired_offset)
    reward = jp.exp(-offset_error / 0.1)
    
    # Return 0 if no detection, otherwise return reward
    return jp.where(no_detection, 0.0, reward)


  def _cost_upright(
      self,
      data  # mjx.Data
  ) -> jax.Array:
    """Penalize deviation from upright posture.
    
    Mirrors MPC "upright" residual.
    
    Args:
      data: MJX state
    
    Returns:
      Negative cost for non-upright orientation
    """
    gravity = self.get_gravity(data)  # (3,)
    
    # Want gravity to point down in body frame (z = -1)
    # So in world frame, torso z-axis should point up (z = +1)
    # gravity[2] should be close to 1.0
    
    upright_error = (1.0 - gravity[2]) ** 2
    
    return -upright_error


  def _cost_height(
      self,
      data,  # mjx.Data
      target_height: float = 1.0
  ) -> jax.Array:
    """Penalize deviation from target torso height.
    
    Mirrors MPC "Height" residual.
    
    Args:
      data: MJX state
      target_height: Desired torso height (meters)
    
    Returns:
      Negative cost for height error
    """
    torso_pos = data.xpos[self._torso_body_id]
    torso_height = torso_pos[2]
    
    height_error = (torso_height - target_height) ** 2
    
    return -height_error


  # ============================================================
  # REWARD SUMMARY UTILITIES
  # ============================================================

  def _get_reward_summary(
      self,
      rewards: Dict[str, jax.Array],
      final_reward: jax.Array
  ) -> Dict[str, jax.Array]:
    """Create summary dict with all reward components for logging.
    
    Args:
      rewards: Reward components
      final_reward: Combined final reward
    
    Returns:
      Dictionary with all reward info
    """
    scales = self._config.reward_config.scales
    
    summary = {
        'reward/total': final_reward,
        'reward/sum': sum(rewards.values()),
    }
    
    # Add scaled individual components
    for k, v in rewards.items():
      summary[f'reward/{k}'] = v * scales[k]
        
    return summary

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
