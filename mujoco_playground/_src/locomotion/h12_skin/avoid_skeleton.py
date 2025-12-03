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
    
    if not self.traj_dir.exists():
      raise FileNotFoundError(f"Trajectory directory not found: {traj_dir}")
    
    # Find all CSV files
    csv_files = sorted(self.traj_dir.glob("episode_*.csv"))
    
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
    
    self.num_trajectories = len(self.trajectories)
    self.trajectory_lengths = np.array(self.trajectory_lengths)
    
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
    
    # Extract robot joint positions (qpos 7-33 = 27 actuated joints)
    # Note: qpos_0-6 are floating base (not needed), qpos_34+ don't exist
    qpos_cols = [f'qpos_{i}' for i in range(7, 34)]
    robot_qpos = df[qpos_cols].values  # (T, 27)
    
    # Extract robot joint velocities (qvel 6-32 = 27 actuated joints)
    # Note: qvel_0-5 are floating base (not needed)
    qvel_cols = [f'qvel_{i}' for i in range(6, 33)]
    robot_qvel = df[qvel_cols].values  # (T, 27)
    
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
        'robot_qpos': robot_qpos,
        'robot_qvel': robot_qvel,
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
    # Sample random trajectory index
    traj_idx = jax.random.randint(rng, (), 0, self.num_trajectories)
    traj_idx_np = int(traj_idx)
    
    # Get trajectory (convert to JAX arrays)
    traj_np = self.trajectories[traj_idx_np]
    traj_jax = {k: jp.array(v) for k, v in traj_np.items()}
    
    return traj_jax
  
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
        'robot_qpos': trajectory['robot_qpos'][step_clamped],  # (27,)
        'robot_qvel': trajectory['robot_qvel'][step_clamped],  # (27,)
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
    return int(trajectory['time'].shape[0])
  
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
        'robot_qpos': (T, 27),
        'robot_qvel': (T, 27),
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
    control_freq_hz: float = 50.0,
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
      from avoid import default_config  # Import from your main file
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
    from avoid import create_trajectory_database  # Import from your main file
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
    from avoid import get_skin_site_ids  # Import from your main file
    skin_site_ids_np = get_skin_site_ids(self._mj_model)
    self._skin_site_ids = jp.array(skin_site_ids_np)
    self._num_skin_sensors = len(skin_site_ids_np)
    print(f"Found {self._num_skin_sensors} skin sensors")
    
    # 4. Store default pose and control ranges
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = self._init_q[7:34]  # Joints 7-33 (27 actuated)
    self._lowers = jp.array(self._mj_model.actuator_ctrlrange[:, 0])
    self._uppers = jp.array(self._mj_model.actuator_ctrlrange[:, 1])
    
    # 5. Get torso body ID for COM tracking
    self._torso_body_id = self._mj_model.body("torso_link").id
    
    # 6. History sampling parameters (for 50Hz history from 200Hz control)
    # With ctrl_dt=0.02 (50Hz control), we want history every 4 steps
    self._history_delta = int(0.08 / self._config.ctrl_dt)  # Every 4 steps at 50Hz
    
    # 7. Capacitance config
    self._cap_eps = self._config.capacitance_config.eps
    self._cap_sensing_radius = self._config.capacitance_config.sensing_radius
    self._cap_collision_threshold = self._config.capacitance_config.collision_threshold
  
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
    
    # Set robot joint positions from reference (qpos 7-33)
    qpos = qpos.at[7:34].set(initial_ref['robot_qpos'])
    
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
    
    # Set robot joint velocities from reference (qvel 6-32)
    qvel = qvel.at[6:33].set(initial_ref['robot_qvel'])
    
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
    from avoid import compute_all_capacitances  # Import from your main file
    capacitances = compute_all_capacitances(
        data,
        self._skin_site_ids,
        self._obstacle_body_id,
        self._obstacle_geom_id,
        self.mjx_model,
        eps=self._cap_eps,
        sensing_radius=self._cap_sensing_radius
    )
    
    # 5. Initialize history buffers
    history_len = self._config.history_len
    qpos_error_history = jp.zeros(history_len * 27)
    qvel_history = jp.zeros(history_len * 27)
    action_history = jp.zeros(history_len * 27)
    capacitance_history = jp.zeros(history_len * self._num_skin_sensors)
    
    # 6. Set up info dict
    info = {
        'rng': rng,
        'current_trajectory': current_trajectory,
        'traj_step': 0,
        'traj_length': traj_length,
        'last_act': jp.zeros(self.mjx_model.nu),
        'last_last_act': jp.zeros(self.mjx_model.nu),
        'motor_targets': jp.zeros(self.mjx_model.nu),
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
    
    # 2. Step physics for n_substeps
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )
    state.info['motor_targets'] = motor_targets
    
    # 3. Increment trajectory step counter
    traj_step = state.info['traj_step'] + 1
    state.info['traj_step'] = traj_step
    
    # 4. Compute capacitances for current state
    from avoid import compute_all_capacitances
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
    history_counter = state.info['history_counter'] + 1
    should_sample = (history_counter % self._history_delta) == 0
    
    # Roll and update histories
    qvel_history = jp.where(
        should_sample,
        jp.roll(state.info['qvel_history'], 27).at[:27].set(data.qvel[6:33]),
        state.info['qvel_history']
    )
    
    qpos_error = data.qpos[7:34] - state.info['motor_targets']
    qpos_error_history = jp.where(
        should_sample,
        jp.roll(state.info['qpos_error_history'], 27).at[:27].set(qpos_error),
        state.info['qpos_error_history']
    )
    
    action_history = jp.where(
        should_sample,
        jp.roll(state.info['action_history'], 27).at[:27].set(action),
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
    pos_rewards, neg_costs = self._get_reward(data, action, state.info, ref_state)
    
    # Scale rewards
    scales = self._config.reward_config.scales
    pos_rewards_scaled = {k: v * scales[k] for k, v in pos_rewards.items()}
    neg_costs_scaled = {k: v * scales[k] for k, v in neg_costs.items()}
    
    # Combine rewards: sum(pos) * exp(0.2 * sum(neg)) * dt
    r_pos = sum(pos_rewards_scaled.values())
    r_neg = jp.exp(0.2 * sum(neg_costs_scaled.values()))
    reward = r_pos * r_neg * self.dt
    
    # 9. Check termination
    done = self._get_termination(data, capacitances)
    
    # 10. Update metrics
    all_rewards = pos_rewards_scaled | neg_costs_scaled
    for k, v in all_rewards.items():
      state.metrics[f'reward/{k}'] = v
    
    # Additional metrics
    state.metrics['collision_detected'] = jp.any(capacitances > self._cap_collision_threshold).astype(jp.float32)
    state.metrics['min_capacitance'] = jp.min(jp.where(capacitances > 0, capacitances, jp.inf))
    state.metrics['tracking_error'] = jp.linalg.norm(data.qpos[7:34] - ref_state['robot_qpos'])
    
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
    """Action space size: 27 actuated joints."""
    return 27
  
  def _compute_obs_size(self) -> tuple:
    """Compute observation sizes for actor and critic.
    
    Returns:
      (actor_obs_size, critic_obs_size)
    """
    history_len = self._config.history_len
    
    # Actor observation size
    actor_proprio_size = (
        3 +  # gyro
        3 +  # gravity
        27 +  # joint_pos_offset
        27 +  # joint_vel
        history_len * 27 +  # qpos_error_history
        history_len * 27 +  # qvel_history
        history_len * 27 +  # action_history
        63 +  # capacitances
        history_len * 63  # capacitance_history
    )
    
    actor_ref_size = (
        27 +  # ref_joint_vel
        3 +   # ref_base_angvel
        27    # ref_joint_pos
    )
    
    actor_obs_size = actor_proprio_size + actor_ref_size
    
    # Critic observation size
    critic_proprio_size = (
        3 +  # base_pos
        4 +  # base_quat
        3 +  # base_linvel
        3 +  # base_angvel
        3 +  # gravity
        history_len * 27 +  # joint_pos_history
        27 +  # joint_vel
        history_len * 27 +  # action_history
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
        27 +  # ref_joint_pos
        27 +  # ref_joint_vel
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
      components['joint_pos_offset'] = obs[idx:idx+27]
      idx += 27
      components['joint_vel'] = obs[idx:idx+27]
      idx += 27
      components['qpos_error_history'] = obs[idx:idx+history_len*27]
      idx += history_len * 27
      components['qvel_history'] = obs[idx:idx+history_len*27]
      idx += history_len * 27
      components['action_history'] = obs[idx:idx+history_len*27]
      idx += history_len * 27
      components['capacitances'] = obs[idx:idx+63]
      idx += 63
      components['capacitance_history'] = obs[idx:idx+history_len*63]
      idx += history_len * 63
      components['ref_joint_vel'] = obs[idx:idx+27]
      idx += 27
      components['ref_base_angvel'] = obs[idx:idx+3]
      idx += 3
      components['ref_joint_pos'] = obs[idx:idx+27]
      idx += 27
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
      components['joint_pos_history'] = obs[idx:idx+history_len*27]
      idx += history_len * 27
      components['joint_vel'] = obs[idx:idx+27]
      idx += 27
      components['action_history'] = obs[idx:idx+history_len*27]
      idx += history_len * 27
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
      components['ref_joint_pos'] = obs[idx:idx+27]
      idx += 27
      components['ref_joint_vel'] = obs[idx:idx+27]
      idx += 27
      components['ref_capacitances'] = obs[idx:idx+63]
      idx += 63
    
    return components
    
  def _get_reward(
      self,
      data,  # mjx.Data
      action: jax.Array,
      info: Dict[str, Any],
      ref_state: Dict[str, jax.Array],
  ) -> Tuple[Dict[str, jax.Array], Dict[str, jax.Array]]:
    """Compute all reward components.
    
    Separates positive rewards (things we want) from negative costs (things we penalize).
    
    Args:
      data: Current MJX state
      action: Current action (27,)
      info: State info dict
      ref_state: Reference state from trajectory at current timestep
    
    Returns:
      pos_rewards: Dictionary of positive reward components
      neg_costs: Dictionary of negative cost components
    """
    # Get capacitances
    capacitances = info['capacitances']
    
    # ============ POSITIVE REWARDS (Tracking + Safety) ============
    
    pos_rewards = {
        # Trajectory tracking rewards
        'joint_pos_tracking': self._reward_joint_pos_tracking(
            data.qpos[7:34], ref_state['robot_qpos']
        ),
        'joint_vel_tracking': self._reward_joint_vel_tracking(
            data.qvel[6:33], ref_state['robot_qvel']
        ),
        'base_vel_tracking': self._reward_base_vel_tracking(
            data.qvel[0:3], ref_state['base_linvel']
        ),
        'base_angvel_tracking': self._reward_base_angvel_tracking(
            data.qvel[3:6], ref_state['base_angvel']
        ),
        'torque_tracking': self._reward_torque_tracking(
            data, ref_state
        ),
        
        # Avoidance rewards
        'clearance_reward': self._reward_clearance(capacitances),
    }
    
    # ============ NEGATIVE COSTS (Penalties + Regularization) ============
    
    neg_costs = {
        # Collision and proximity penalties
        'collision_penalty': self._cost_collision(capacitances),
        'proximity_penalty': self._cost_proximity(capacitances),
        
        # Regularization costs
        'action_rate': self._cost_action_rate(action, info['last_act']),
        'torque': self._cost_torque(data),
        'energy': self._cost_energy(data),
        'joint_limit': self._cost_joint_limit(data.qpos[7:34]),
    }
    
    return pos_rewards, neg_costs


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
      qpos: Current joint positions (27,)
      ref_qpos: Reference joint positions (27,)
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
      qvel: Current joint velocities (27,)
      ref_qvel: Reference joint velocities (27,)
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
    
    # Get current actuator forces (torques)
    current_torques = data.qfrc_actuator[6:33]  # (27,) actuated joints
    ref_torques = ref_state['joint_torques']
    
    error = current_torques - ref_torques
    squared_error = jp.sum(error ** 2)
    return jp.exp(-squared_error / sigma)


  def _reward_clearance(
      self,
      capacitances: jax.Array,
      safe_threshold: float = 2.0
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
    
    # Return -1.0 if collision, 0.0 otherwise
    return jp.where(collision_detected, -1.0, 0.0)


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
    proximity = jp.where(in_range, capacitances ** 2, 0.0)
    total_proximity = jp.sum(proximity)
    
    # Normalize by number of sensors for consistent scale
    avg_proximity = total_proximity / jp.float32(capacitances.shape[0])
    
    return -avg_proximity


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
      action: Current action (27,)
      last_action: Previous action (27,)
    
    Returns:
      Negative cost proportional to action change
    """
    delta = action - last_action
    return -jp.sum(delta ** 2)


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
    # Get actuator forces (torques)
    torques = data.qfrc_actuator[6:33]  # (27,) actuated joints
    return -jp.sum(torques ** 2)


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
    torques = data.qfrc_actuator[6:33]  # (27,)
    qvel = data.qvel[6:33]  # (27,)
    power = torques * qvel
    return -jp.sum(jp.abs(power))


  def _cost_joint_limit(
      self,
      qpos: jax.Array,
      margin: float = 0.1
  ) -> jax.Array:
    """Penalize approaching joint limits.
    
    Args:
      qpos: Joint positions (27,)
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
    3. Collision detected (capacitance > threshold)
    4. Robot moves too far from origin (optional safety)
    
    Args:
      data: MJX state
      capacitances: (63,) capacitance readings
    
    Returns:
      Boolean (as float): 1.0 if should terminate, 0.0 otherwise
    """
    # 1. Check joint limits
    joint_angles = data.qpos[7:34]
    joint_limit_exceed = jp.any(joint_angles < self._lowers)
    joint_limit_exceed |= jp.any(joint_angles > self._uppers)
    
    # 2. Check if robot falls (gravity sensor z < threshold)
    gravity = self.get_gravity(data)
    fall_termination = gravity[2] < 0.85  # Torso tilted significantly
    
    # 3. Check collision
    collision_detected = jp.any(capacitances > self._cap_collision_threshold)
    
    # 4. Check if robot moved too far (optional, safety boundary)
    base_pos = data.qpos[0:3]
    distance_from_origin = jp.linalg.norm(base_pos[:2])  # xy distance
    too_far = distance_from_origin > 5.0  # 5m safety radius
    
    # Combine conditions
    terminate = joint_limit_exceed | fall_termination | collision_detected | too_far
    
    # Respect early_termination config flag
    terminate = jp.where(
        self._config.early_termination,
        terminate,
        joint_limit_exceed  # Always terminate on joint limits
    )
    
    return terminate.astype(jp.float32)


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
    from avoid import compute_obstacle_centroid
    
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
      pos_rewards: Dict[str, jax.Array],
      neg_costs: Dict[str, jax.Array],
      final_reward: jax.Array
  ) -> Dict[str, jax.Array]:
    """Create summary dict with all reward components for logging.
    
    Args:
      pos_rewards: Positive reward components
      neg_costs: Negative cost components
      final_reward: Combined final reward
    
    Returns:
      Dictionary with all reward info
    """
    scales = self._config.reward_config.scales
    
    summary = {
        'reward/total': final_reward,
        'reward/positive_sum': sum(pos_rewards.values()),
        'reward/negative_sum': sum(neg_costs.values()),
    }
    
    # Add scaled individual components
    for k, v in pos_rewards.items():
      summary[f'reward/{k}'] = v * scales[k]
    
    for k, v in neg_costs.items():
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
