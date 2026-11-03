from __future__ import annotations

import logging
import time
from typing import TypedDict

import msgpack
import numpy as np
import zmq

_log = logging.getLogger(__name__)


class BambooConnectionError(Exception):
    """Raised when connection to bamboo control node fails."""

    pass


class BambooTimeoutError(Exception):
    """Raised when operation times out."""

    pass


class BambooGripperError(Exception):
    """Raised when gripper operation fails."""

    pass


class JointStates(TypedDict, total=False):
    """Type definition for joint states dictionary.

    The keys after ``gripper_is_moving`` are present only when the server
    reports them (servers with the ``external_wrench`` / ``guide_mode``
    features); older servers simply leave them out.
    """

    ee_pose: list[list[float]]
    qpos: list[float]
    dq: list[float]
    tau_J: list[float]
    time_sec: float
    gripper_state: float
    gripper_is_grasped: bool
    gripper_is_moving: bool
    ext_wrench: list[float]
    tau_ext: list[float]
    loop_mode: str
    guide_phase: str
    guide_hold_reason: str
    guide_level: float
    guide_lead: float
    guide_ratchet: bool
    contact_scale: float
    contact_push: float
    guard_trip: str


LOOP_MODES = {0: "idle", 1: "trajectory", 2: "stream", 3: "guide"}
GUIDE_PHASES = {0: "free", 1: "hold_damp", 2: "hold_ramp", 3: "hold", 4: "release"}
HOLD_REASONS = {
    0: "none",
    1: "watchdog",
    2: "gel_invalid",
    3: "gel_limit",
    4: "joint_limit",
    5: "stopping",
    6: "idle",
}
# Why the last stream yielded (Bamboo's contact guard, or a client's request).
GUARD_TRIPS = {0: "none", 1: "sustained", 2: "hard", 3: "invalid", 4: "requested"}


class BambooFrankaClient:
    """Client for communicating with the bamboo control node."""

    def __init__(
        self,
        control_port: int = 5555,
        server_ip: str = "localhost",
        gripper_port: int = 5559,
        enable_gripper: bool = True,
        gripper_type: str = "robotiq",
    ):
        """Initialize Bamboo Franka Client.

        Args:
            control_port: Port of the bamboo control node (default: 5555)
            server_ip: IP address of the bamboo control node server
            gripper_port: ZMQ port of the gripper server (default: 5559)
            enable_gripper: Whether to enable gripper commands
            gripper_type: Type of gripper - "robotiq" or "franka" (default: "robotiq")
        """
        self.control_port = control_port
        self.server_ip = server_ip
        self.gripper_port = gripper_port
        self.enable_gripper = enable_gripper
        self.gripper_type = gripper_type
        self._streaming = False
        self._guiding = False

        # Validate gripper type
        if gripper_type not in ["robotiq", "franka"]:
            raise ValueError(f"Invalid gripper_type '{gripper_type}'. Must be 'robotiq' or 'franka'")

        # Set up ZMQ context and control socket
        self.zmq_context = zmq.Context()

        # Set up control socket (REQ socket for request-response)
        self.control_socket = self.zmq_context.socket(zmq.REQ)
        self.control_socket.connect(f"tcp://{self.server_ip}:{self.control_port}")
        self.control_socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
        _log.debug(f"Control client connected to {self.server_ip}:{self.control_port}")

        self.gripper_socket = None

        # For Robotiq, create separate gripper socket to gripper_server
        # For Franka, gripper commands go through control_socket to control_node
        if enable_gripper and gripper_type == "robotiq":
            # Set up ZMQ socket for gripper commands (REQ socket for request-response)
            self.gripper_socket = self.zmq_context.socket(zmq.REQ)
            self.gripper_socket.connect(f"tcp://{self.server_ip}:{gripper_port}")
            self.gripper_socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
            _log.debug(f"Gripper client connected to {self.server_ip}:{gripper_port}")

        # Test connection by trying to receive a state message
        self._test_connection()

    def _recreate_control_socket(self) -> None:
        """Recreate the control socket after it becomes corrupted.

        This is necessary when ZMQ REQ socket gets into a bad state,
        typically after connection issues or pattern violations.
        """
        _log.warning("Recreating control socket due to socket error")

        # Close the old socket if it exists
        if hasattr(self, "control_socket") and self.control_socket is not None:
            try:
                self.control_socket.close()
            except Exception as e:
                _log.warning(f"Failed to close corrupted control socket: {e}")

        # Try to create a new socket with the existing context
        try:
            self.control_socket = self.zmq_context.socket(zmq.REQ)
        except zmq.ZMQError as e:
            # If socket creation fails, the context itself may be corrupted
            # Recreate the entire context
            _log.warning(f"Failed to create socket from existing context ({e}). Recreating ZMQ context...")

            # Close gripper socket if it exists
            if hasattr(self, "gripper_socket") and self.gripper_socket is not None:
                try:
                    self.gripper_socket.close()
                except Exception as e2:
                    _log.warning(f"Failed to close gripper socket: {e2}")

            # Terminate old context
            try:
                self.zmq_context.term()
            except Exception as e2:
                _log.warning(f"Failed to terminate ZMQ context: {e2}")

            # Create new context and sockets
            self.zmq_context = zmq.Context()
            self.control_socket = self.zmq_context.socket(zmq.REQ)

            # Recreate gripper socket if it was enabled and gripper type is robotiq
            if self.enable_gripper and self.gripper_type == "robotiq":
                self.gripper_socket = self.zmq_context.socket(zmq.REQ)
                self.gripper_socket.connect(f"tcp://{self.server_ip}:{self.gripper_port}")
                self.gripper_socket.setsockopt(zmq.RCVTIMEO, 5000)
                _log.debug(f"Gripper socket recreated at {self.server_ip}:{self.gripper_port}")

        self.control_socket.connect(f"tcp://{self.server_ip}:{self.control_port}")
        self.control_socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
        _log.debug(f"Control socket reconnected to {self.server_ip}:{self.control_port}")

    def _test_connection(self) -> None:
        """Test connection to the bamboo control node.

        Raises:
            BambooConnectionError: If connection test fails
        """
        try:
            # Try to receive a state message to verify connection
            _log.debug(f"Establishing connection to bamboo control node at {self.server_ip}:{self.control_port}...")
            self._get_latest_state()
            _log.info(f"Successfully connected to bamboo control node at {self.server_ip}:{self.control_port}")
        except Exception as e:
            raise BambooConnectionError(
                f"Could not connect to bamboo control node at {self.server_ip}:{self.control_port}. "
                f"Make sure the bamboo control node is running. Error: {e}"
            ) from e

    def _get_latest_state(self, max_retries: int = 3) -> dict:
        """Get the latest robot state from the bamboo control node.

        Args:
            max_retries: Maximum number of retry attempts if socket becomes corrupted (default: 3)

        Returns:
            dict: The latest robot state containing q, dq, tau_J, O_T_EE, time_sec

        Raises:
            BambooConnectionError: If server returns error or max retries exceeded
            BambooTimeoutError: If state read times out
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                # Create request
                request = {"command": "get_robot_state"}

                # Send request
                self.control_socket.send(msgpack.packb(request))

                # Receive response
                response_data = self.control_socket.recv()
                response = msgpack.unpackb(response_data, raw=False)

                if not response.get("success", False):
                    raise BambooConnectionError(f"Failed to get robot state: {response.get('error', 'Unknown error')}")

                return response["data"]

            except zmq.Again:
                raise BambooTimeoutError("Timeout waiting for robot state response") from None

            except zmq.ZMQError as e:
                # Handle socket corruption errors (e.g., "Socket operation on non-socket")
                last_error = e
                if attempt < max_retries - 1:
                    _log.warning(f"ZMQ socket error on attempt {attempt + 1}/{max_retries}: {e}. Recreating socket...")
                    self._recreate_control_socket()
                    # Retry the operation with the new socket
                    continue

        # Max retries exceeded, raise error
        raise BambooConnectionError(
            f"Failed to get robot state after {max_retries} attempts due to socket errors: {last_error}"
        ) from last_error

    def get_joint_states(self) -> JointStates:
        """Get current robot joint states.

        Returns:
            Dict with 'ee_pose', 'qpos', 'dq', 'tau_J', 'time_sec', 'gripper_state'

        Raises:
            BambooConnectionError: If connection to controller fails
            BambooTimeoutError: If state read times out
            BambooGripperError: If gripper state read fails
        """
        # Get latest state from bamboo control node
        state_msg = self._get_latest_state()

        # Extract joint positions
        qpos = list(state_msg["q"])  # Convert to list for JSON serialization

        # Extract joint velocities
        dq = list(state_msg["dq"])

        # Extract joint torques
        tau_J = list(state_msg["tau_J"])

        # Extract timestamp
        time_sec = float(state_msg["time_sec"])

        # Extract end-effector pose (4x4 transformation matrix)
        # Convert flat array to 4x4 matrix (column-major)
        ee_pose_flat = list(state_msg["O_T_EE"])
        ee_pose_matrix = np.array(ee_pose_flat).reshape(4, 4)
        # Transpose from column-major to row-major
        ee_pose = ee_pose_matrix.T.tolist()

        # Get gripper state if available, otherwise use default
        if self.enable_gripper:
            gripper_result = self.get_gripper_state()
            if not gripper_result.get("success"):
                raise BambooGripperError(f"Failed to get gripper state: {gripper_result.get('error', 'Unknown error')}")
            gripper_state = gripper_result["state"]["width"]
            gripper_is_grasped = gripper_result["state"]["is_grasped"]
            gripper_is_moving = gripper_result["state"]["is_moving"]
        else:
            gripper_state = 0.0  # No gripper enabled
            gripper_is_grasped = False
            gripper_is_moving = False

        states: JointStates = {
            "ee_pose": ee_pose,
            "qpos": qpos,
            "dq": dq,
            "tau_J": tau_J,
            "time_sec": time_sec,
            "gripper_state": gripper_state,
            "gripper_is_grasped": gripper_is_grasped,
            "gripper_is_moving": gripper_is_moving,
        }
        # Newer servers add these; read them only when present.
        if "O_F_ext_hat_K" in state_msg:
            states["ext_wrench"] = list(state_msg["O_F_ext_hat_K"])
        if "tau_ext_hat_filtered" in state_msg:
            states["tau_ext"] = list(state_msg["tau_ext_hat_filtered"])
        if "loop_mode" in state_msg:
            states["loop_mode"] = LOOP_MODES.get(int(state_msg["loop_mode"]), "unknown")
        if "guide_phase" in state_msg:
            states["guide_phase"] = GUIDE_PHASES.get(int(state_msg["guide_phase"]), "unknown")
            states["guide_hold_reason"] = HOLD_REASONS.get(int(state_msg.get("guide_reason", 0)), "unknown")
            states["guide_level"] = float(state_msg.get("guide_level", 0.0))
        if "guide_lead" in state_msg:
            states["guide_lead"] = float(state_msg["guide_lead"])
            states["guide_ratchet"] = bool(state_msg.get("guide_ratchet", False))
        if "contact_scale" in state_msg:
            states["contact_scale"] = float(state_msg["contact_scale"])
            states["contact_push"] = float(state_msg.get("contact_push", 0.0))
            states["guard_trip"] = GUARD_TRIPS.get(int(state_msg.get("guard_trip", 0)), "unknown")
        return states

    def close(self) -> None:
        """Clean up ZMQ resources."""
        if getattr(self, "_streaming", False):
            try:
                self.stop_streaming()
            except Exception as e:
                _log.error(f"Failed to stop stream cleanly; server watchdog will brake the robot: {e}")
        if getattr(self, "_guiding", False):
            try:
                self.stop_guiding()
            except Exception as e:
                _log.error(f"Failed to stop guiding cleanly; the server watchdog will hold the arm: {e}")
        if hasattr(self, "control_socket") and self.control_socket is not None:
            self.control_socket.close()
        if hasattr(self, "gripper_socket") and self.gripper_socket is not None:
            self.gripper_socket.close()
        if hasattr(self, "zmq_context") and self.zmq_context is not None:
            self.zmq_context.term()

    def __enter__(self) -> BambooFrankaClient:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Context manager exit."""
        self.close()

    def get_joint_positions(self) -> list[float]:
        return self.get_joint_states()["qpos"]

    def _send_panda_hand_command(self, command: dict, timeout_ms: int = 5000) -> dict:
        """Send a command to the control node.

        Args:
            command: Dict with command to send
            timeout_ms: Timeout in milliseconds

        Returns:
            Dict with response from control node

        Raises:
            BambooTimeoutError: If command times out
        """
        try:
            # Save old timeout
            old_timeout = self.control_socket.getsockopt(zmq.RCVTIMEO)
            self.control_socket.setsockopt(zmq.RCVTIMEO, timeout_ms)

            try:
                # Send command
                self.control_socket.send(msgpack.packb(command))

                # Receive response
                response_data = self.control_socket.recv()
                response = msgpack.unpackb(response_data, raw=False)

                return response  # type: ignore[no-any-return]
            finally:
                # Restore timeout
                self.control_socket.setsockopt(zmq.RCVTIMEO, old_timeout)

        except zmq.Again:
            raise BambooTimeoutError("Timeout waiting for control node response") from None

    def get_capabilities(self) -> dict:
        """Return capabilities advertised by the control node.

        Older Bamboo servers return an unsuccessful response for this command;
        callers can use :meth:`supports_streaming` for a boolean check.
        """

        return self._send_panda_hand_command({"command": "get_capabilities"})

    def supports_streaming(self) -> bool:
        """Return whether persistent joint-velocity streaming is available."""

        response = self.get_capabilities()
        return bool(response.get("success", False) and "stream_joint_velocity" in response.get("features", []))

    def start_streaming(
        self,
        watchdog_ms: int = 250,
        max_joint_velocity: float = 0.35,
        max_joint_acceleration: float = 1.5,
    ) -> dict:
        """Start a persistent joint-impedance control session.

        The server ramps velocity commands at ``max_joint_acceleration`` and
        commands zero velocity if no update arrives within ``watchdog_ms``.
        Server-side hard limits cap all three values.
        """

        if not 50 <= watchdog_ms <= 500:
            raise ValueError("watchdog_ms must be between 50 and 500")
        if max_joint_velocity <= 0:
            raise ValueError("max_joint_velocity must be positive")
        if max_joint_acceleration <= 0:
            raise ValueError("max_joint_acceleration must be positive")
        response = self._send_panda_hand_command(
            {
                "command": "start_stream",
                "data": {
                    "watchdog_ms": watchdog_ms,
                    "max_joint_velocity": max_joint_velocity,
                    "max_joint_acceleration": max_joint_acceleration,
                },
            }
        )
        if response.get("success", False):
            self._streaming = True
        return response

    def stream_joint_velocity(self, velocity: np.ndarray | list[float]) -> dict:
        """Update the seven joint velocities of an active stream, in rad/s."""

        values = np.asarray(velocity, dtype=float)
        if values.shape != (7,):
            raise ValueError(f"velocity must have shape (7,), got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("velocity values must be finite")
        if not self._streaming:
            raise RuntimeError("Streaming has not been started by this client")
        return self._send_panda_hand_command(
            {
                "command": "stream_joint_velocity",
                "data": {"velocity": values.tolist()},
            }
        )

    def stop_streaming(self) -> dict:
        """Ramp an active stream to rest and end its impedance session."""

        response = self._send_panda_hand_command({"command": "stop_stream"}, timeout_ms=5000)
        if response.get("success", False):
            self._streaming = False
        return response

    def supports(self, feature: str) -> bool:
        """Return whether the control node advertises ``feature``."""

        response = self.get_capabilities()
        return bool(response.get("success", False) and feature in response.get("features", []))

    def start_guiding(
        self,
        watchdog_ms: int = 250,
        max_joint_velocity: float = 0.35,
        joint_damping: np.ndarray | list[float] | None = None,
        contact_point: np.ndarray | list[float] | None = None,
        wall_damping: float | None = None,
        idle_hold_s: float | None = None,
        max_start_force: float | None = None,
    ) -> dict:
        """Start hand guiding: the arm floats under gravity compensation.

        The arm moves freely under light damping. Joints resist above
        ``max_joint_velocity`` (rad/s) and near their limits. If no
        :meth:`update_guiding` arrives within ``watchdog_ms``, the arm damps to
        a stop and holds until :meth:`resume_guiding`. ``contact_point`` (m,
        end-effector frame) is where the gel wall acts; ``wall_damping`` is its
        damping at level 1 (N s/m). After ``idle_hold_s`` of stillness (0 turns
        it off) the arm holds until pushed. The server refuses to start while
        Franka's external force estimate exceeds ``max_start_force`` (N; 0
        disables), which catches an uncompensated payload.
        """

        if not 50 <= watchdog_ms <= 500:
            raise ValueError("watchdog_ms must be between 50 and 500")
        if max_joint_velocity <= 0:
            raise ValueError("max_joint_velocity must be positive")
        data: dict = {"watchdog_ms": int(watchdog_ms), "max_joint_velocity": float(max_joint_velocity)}
        if joint_damping is not None:
            damping = np.asarray(joint_damping, dtype=float)
            if damping.shape != (7,) or not np.all(np.isfinite(damping)) or np.any(damping < 0):
                raise ValueError("joint_damping must be seven finite, non-negative values")
            data["joint_damping"] = damping.tolist()
        if contact_point is not None:
            point = np.asarray(contact_point, dtype=float)
            if point.shape != (3,) or not np.all(np.isfinite(point)):
                raise ValueError("contact_point must be three finite values")
            data["contact_point"] = point.tolist()
        if wall_damping is not None:
            if not np.isfinite(wall_damping) or wall_damping < 0:
                raise ValueError("wall_damping must be finite and non-negative")
            data["wall_damping"] = float(wall_damping)
        if idle_hold_s is not None:
            if not np.isfinite(idle_hold_s) or idle_hold_s < 0:
                raise ValueError("idle_hold_s must be finite and non-negative")
            data["idle_hold_s"] = float(idle_hold_s)
        if max_start_force is not None:
            if not np.isfinite(max_start_force) or max_start_force < 0:
                raise ValueError("max_start_force must be finite and non-negative")
            data["max_start_force"] = float(max_start_force)
        response = self._send_panda_hand_command({"command": "start_guide", "data": data})
        if response.get("success", False):
            self._guiding = True
        return response

    def update_guiding(
        self,
        level: float = 0.0,
        valid: bool = True,
        direction: np.ndarray | list[float] | None = None,
        seq: int | None = None,
    ) -> dict:
        """Keep guiding alive and set the gel wall.

        ``level`` (0 to 1) scales the wall; 1 or ``valid=False`` makes the arm
        hold. ``direction`` (end-effector frame) is the motion the wall resists;
        leave it out to damp all motion. ``seq`` must increase between updates
        for the watchdog to count them; it defaults to this host's monotonic
        clock, so send an update only when there is something new to say.
        """

        data: dict = {
            "level": float(level),
            "valid": bool(valid),
            "seq": int(time.monotonic_ns() if seq is None else seq),
        }
        if direction is not None:
            vector = np.asarray(direction, dtype=float)
            if vector.shape != (3,):
                raise ValueError("direction must have three values")
            data["direction"] = vector.tolist()
        return self._send_panda_hand_command({"command": "guide_update", "data": data})

    def resume_guiding(self) -> dict:
        """Ask the arm to leave a hold. The server refuses unless it is safe."""

        return self._send_panda_hand_command({"command": "guide_resume"})

    def stop_guiding(self) -> dict:
        """Bring the arm to a hold and end hand guiding."""

        response = self._send_panda_hand_command({"command": "stop_guide"}, timeout_ms=5000)
        if response.get("success", False):
            self._guiding = False
        return response

    def set_load(
        self,
        mass: float,
        center_of_mass: np.ndarray | list[float],
        inertia: np.ndarray | list[float] | None = None,
    ) -> dict:
        """Tell Franka about a payload beyond Desk's end effector (kg, m in the flange frame)."""

        com = np.asarray(center_of_mass, dtype=float)
        if not np.isfinite(mass) or mass < 0 or com.shape != (3,) or not np.all(np.isfinite(com)):
            raise ValueError("mass must be non-negative and center_of_mass three finite values")
        data: dict = {"mass": float(mass), "center_of_mass": com.tolist()}
        if inertia is not None:
            matrix = np.asarray(inertia, dtype=float).reshape(-1)
            if matrix.shape != (9,) or not np.all(np.isfinite(matrix)):
                raise ValueError("inertia must be nine finite values")
            data["inertia"] = matrix.tolist()
        return self._send_panda_hand_command({"command": "set_load", "data": data})

    def recover_errors(self) -> dict:
        """Clear a Franka reflex so a new control loop can start."""

        return self._send_panda_hand_command({"command": "recover_errors"}, timeout_ms=5000)

    def set_contact_limits(
        self,
        max_force_n: float = 0.0,
        contact_point: np.ndarray | list[float] | None = None,
        max_point_speed: float = 0.0,
        guard_force_n: float = 0.0,
        guard_hard_force_n: float = 0.0,
        guard_impulse_ns: float = 0.05,
        guard_on_invalid: bool = True,
        yield_time: float = 0.2,
        collision_force_n: float = 0.0,
        shadow: bool = False,
    ) -> dict:
        """Limit how hard every following stream can load the tool (needs ``contact_limits``).

        ``max_force_n`` caps the force the stream could push with at
        ``contact_point`` (m, end-effector frame) if contact stopped it there;
        ``shadow`` reports that cap in the state without applying it.
        ``max_point_speed`` (m/s) slows the contact point. The guard watches
        Franka's external force estimate: above ``guard_force_n`` for
        ``guard_impulse_ns`` of excess, or at once above
        ``guard_hard_force_n`` (default 1.5x), the arm goes soft for
        ``yield_time`` s and the stream ends with the reason. The estimate
        includes anything the gripper lifts, so set the guard above that.
        ``collision_force_n`` lowers Franka's own reflex threshold as a last
        resort; guide mode restores it. 0 leaves a limit off. Only while no
        control loop runs; the reply echoes what the server applied.
        """

        point = np.zeros(3) if contact_point is None else np.asarray(contact_point, dtype=float)
        values = [
            max_force_n,
            max_point_speed,
            guard_force_n,
            guard_hard_force_n,
            guard_impulse_ns,
            yield_time,
            collision_force_n,
        ]
        if point.shape != (3,) or not np.all(np.isfinite(point)) or not np.all(np.isfinite(values)):
            raise ValueError("contact limits must be finite and contact_point three values")
        if min(values) < 0:
            raise ValueError("contact limits must not be negative")
        data = {
            "max_force_n": float(max_force_n),
            "contact_point": point.tolist(),
            "max_point_speed": float(max_point_speed),
            "guard_force_n": float(guard_force_n),
            "guard_hard_force_n": float(guard_hard_force_n),
            "guard_impulse_ns": float(guard_impulse_ns),
            "guard_on_invalid": bool(guard_on_invalid),
            "yield_time": float(yield_time),
            "collision_force_n": float(collision_force_n),
            "shadow": bool(shadow),
        }
        return self._send_panda_hand_command({"command": "set_contact_limits", "data": data})

    def clear_contact_limits(self) -> dict:
        """Remove contact limits and restore Bamboo's collision thresholds."""

        return self._send_panda_hand_command({"command": "clear_contact_limits"})

    def yield_streaming(self, reason: str = "") -> dict:
        """Make an active stream stop pushing (damping only), then end with ``reason``."""

        return self._send_panda_hand_command({"command": "yield_stream", "data": {"reason": str(reason)}})

    def _send_robotiq_command(self, command: dict) -> dict:
        """Send a command to the gripper server.

        Args:
            command: Dict with command to send

        Returns:
            Dict with response from gripper server

        Raises:
            BambooGripperError: If gripper not enabled
            BambooTimeoutError: If gripper command times out
        """
        if not self.enable_gripper or self.gripper_socket is None:
            raise BambooGripperError("Gripper not enabled or not connected")

        try:
            # Send command
            self.gripper_socket.send(msgpack.packb(command))

            # Receive response
            response_data = self.gripper_socket.recv()
            response = msgpack.unpackb(response_data, raw=False)

            return response  # type: ignore[no-any-return]

        except zmq.Again:
            raise BambooTimeoutError("Timeout waiting for gripper server response") from None

    def _validate_gains(
        self, joint_confs: np.ndarray, gains: np.ndarray | None, label: str
    ) -> list[list[float]] | None:
        gains_per_waypoint = None
        if gains is not None:
            if not isinstance(gains, np.ndarray):
                raise TypeError(f"{label} must be a numpy array, got {type(gains)}")

            # Validate non-negative
            if np.any(gains < 0):
                raise ValueError(f"All {label} values must be >= 0, got min value {gains.min()}")

            if gains.ndim == 1:
                # Single array of 7 values for all waypoints
                if gains.shape[0] != 7:
                    raise ValueError(f"{label} must have shape (7,), got {gains.shape}")
                # Broadcast to all waypoints - convert to list for each waypoint
                gains_per_waypoint = [gains.tolist()] * joint_confs.shape[0]
            elif gains.ndim == 2:
                # Per-waypoint gains
                if gains.shape[0] != joint_confs.shape[0]:
                    raise ValueError(
                        f"{label} must have same number of rows as joint_confs "
                        f"({joint_confs.shape[0]}), got {gains.shape[0]}"
                    )
                if gains.shape[1] != 7:
                    raise ValueError(f"{label} must have 7 columns, got {gains.shape[1]}")
                # Convert each row to list
                gains_per_waypoint = [gains[i].tolist() for i in range(gains.shape[0])]
            else:
                raise ValueError(f"{label} must be 1D or 2D array, got {gains.ndim}D")

        return gains_per_waypoint

    def execute_joint_impedance_path(
        self,
        joint_confs: np.ndarray,
        joint_vels: np.ndarray | None = None,
        durations: list | None = None,
        default_duration: float = 0.5,
        kp: np.ndarray | None = None,
        kd: np.ndarray | None = None,
    ) -> dict[str, bool | str]:
        """Execute joint impedance trajectory and wait for completion.

        Args:
            joint_confs: (n, 7) array of joint configurations
            joint_vels: (n, 7) array of of joint velocities for each waypoint
            durations: Optional list of durations (in seconds) for each waypoint.
                      If None, all waypoints use default_duration.
                      If shorter than joint_confs, remaining waypoints use default_duration.
            default_duration: Default duration in seconds for waypoints (default: 1.0s)
            kp: Optional stiffness gains. Can be:
                - None: Use controller's tuned defaults for all waypoints
                - np.ndarray of shape (7,): Use same custom gains for all waypoints
                - np.ndarray of shape (n, 7): Use different gains per waypoint
            kd: Optional damping gains. Can be:
                - None: Use controller's tuned defaults for all waypoints
                - np.ndarray of shape (7,): Use same custom gains for all waypoints
                - np.ndarray of shape (n, 7): Use different gains per waypoint

        Returns:
            Dict with 'success' (bool) and 'error' (str) if failed
        """
        try:
            # Validate joint_confs shape
            if joint_confs.ndim != 2:
                raise ValueError(f"joint_confs must be 2D array, got {joint_confs.ndim}D")
            if joint_confs.shape[1] != 7:
                raise ValueError(f"joint_confs must have 7 columns, got {joint_confs.shape[1]}")

            _log.debug(f"Executing {joint_confs.shape[0]} joint waypoints")

            # Validate kp and kd
            kp_per_waypoint = self._validate_gains(joint_confs, kp, "kp")
            kd_per_waypoint = self._validate_gains(joint_confs, kd, "kd")

            # Validate joint_vels parameter
            if joint_vels is None:
                joint_vels = np.array([[0, 0, 0, 0, 0, 0, 0]] * joint_confs.shape[0])
            else:
                if joint_vels.ndim != 2:
                    raise ValueError(f"joint_vels must be 2D array, got {joint_vels.ndim}D")
                if joint_vels.shape[1] != 7:
                    raise ValueError(f"joint_vels must have 7 columns, got {joint_vels.shape[1]}")
                if joint_vels.shape[0] != joint_confs.shape[0]:
                    raise ValueError(
                        f"joint_vels rows ({joint_vels.shape[0]}) must match joint_confs rows ({joint_confs.shape[0]})"
                    )

            # Build trajectory request with all waypoints
            waypoints = []
            total_duration = 0.0

            # Create each waypoint as a TimedWaypoint
            for i, joint_conf in enumerate(joint_confs):
                joint_vel = joint_vels[i]

                # Get waypoint duration
                waypoint_duration = durations[i] if (durations is not None and i < len(durations)) else default_duration
                if waypoint_duration <= 0:
                    waypoint_duration = default_duration
                total_duration += waypoint_duration

                # Create timed waypoint
                waypoint = {
                    "q_goal": joint_conf.tolist(),
                    "velocity": joint_vel.tolist(),
                    "duration": waypoint_duration,
                }

                if kp_per_waypoint is not None:
                    waypoint["kp"] = kp_per_waypoint[i]
                if kd_per_waypoint is not None:
                    waypoint["kd"] = kd_per_waypoint[i]

                waypoints.append(waypoint)

            # Create trajectory request
            trajectory_request = {"waypoints": waypoints, "default_duration": default_duration, "default_velocity": []}

            # Create full request
            request = {"command": "execute_trajectory", "data": trajectory_request}

            # Calculate appropriate timeout (total duration + buffer for settling + safety margin)
            # Add 2 seconds for robot settling time plus 50% safety margin
            trajectory_timeout_ms = int((total_duration + 2.0) * 1.5 * 1000)
            # Minimum timeout of 100ms
            trajectory_timeout_ms = max(trajectory_timeout_ms, 100)

            # Temporarily set socket timeout for this trajectory
            old_timeout = self.control_socket.getsockopt(zmq.RCVTIMEO)
            self.control_socket.setsockopt(zmq.RCVTIMEO, trajectory_timeout_ms)
            _log.debug(
                f"Set trajectory timeout to {trajectory_timeout_ms / 1000.0:.1f}s for {total_duration:.1f}s trajectory"
            )

            try:
                # Send trajectory to control node and wait for completion
                _log.debug("Sending trajectory to control node...")
                self.control_socket.send(msgpack.packb(request))

                # Receive response (this will block until trajectory completes)
                response_data = self.control_socket.recv()
                response = msgpack.unpackb(response_data, raw=False)
            finally:
                # Restore original timeout
                self.control_socket.setsockopt(zmq.RCVTIMEO, old_timeout)

            if response.get("success", False):
                _log.debug("Trajectory completed successfully")
                return {"success": True}
            else:
                error_msg = response.get("error", "Trajectory execution failed")
                _log.error(
                    f"Trajectory failed: {error_msg}. \n"
                    f" You may have to restart the controller node on the machine connected to the robot."
                )
                return {"success": False, "error": error_msg}

        except zmq.Again:
            _log.error("Timeout waiting for trajectory completion")
            return {"success": False, "error": "Timeout waiting for trajectory completion"}
        except ValueError as e:
            # Validation errors from parameter checks
            error_msg = f"Invalid trajectory parameters: {e}"
            _log.error(error_msg)
            return {"success": False, "error": error_msg}

    def open_gripper(self, speed: float = 0.05, force: float = 0.1, blocking: bool = True) -> dict:
        """Open the gripper.

        Args:
            speed: Gripper opening speed (0.0 to 1.0 for robotiq, 0.01 to 0.1 for panda hand)
            force: Gripper force (Robotiq only)
            blocking: Whether to block until gripper finishes (Robotiq only, Franka always blocks)

        Returns:
            Dict with response from gripper

        Raises:
            BambooGripperError: If gripper not enabled or command fails
            BambooTimeoutError: If gripper command times out
        """
        if not self.enable_gripper:
            raise BambooGripperError("Gripper not enabled")

        if self.gripper_type == "franka":
            command = {
                "command": "open_gripper",
                "width": 0.08,  # Max width for Franka Hand
                "speed": speed,
            }
            return self._send_panda_hand_command(command)
        else:
            command = {
                "command": "open_gripper",
                "width": 0.085,  # Max width for Robotiq 2F-85
                "speed": speed,
                "force": force,
                "blocking": blocking,
            }
            return self._send_robotiq_command(command)

    def close_gripper(self, speed: float = 0.05, force: float = 0.8, blocking: bool = True) -> dict:
        """Close the gripper.

        Args:
            speed: Gripper closing speed (0.0 to 1.0 for robotiq, 0.01 to 0.1 for panda hand)
            force: Gripper force (0.0 to 1.0, mapped to 5-70N for Franka)
            blocking: Whether to block until gripper finishes (Robotiq only, Franka always blocks)

        Returns:
            Dict with response from gripper

        Raises:
            BambooGripperError: If gripper not enabled or command fails
            BambooTimeoutError: If gripper command times out
        """
        if not self.enable_gripper:
            raise BambooGripperError("Gripper not enabled")

        if self.gripper_type == "franka":
            command = {
                "command": "close_gripper",
                "speed": speed,
                "force": force,  # Will be converted to Newtons in C++ (5-70N)
            }
            return self._send_panda_hand_command(command)
        else:
            command = {
                "command": "close_gripper",
                "speed": speed,
                "force": force,
                "blocking": blocking,
            }
            return self._send_robotiq_command(command)

    def get_gripper_state(self) -> dict:
        """Get current gripper state.

        Returns:
            Dict with 'success' and 'state' from gripper

        Raises:
            BambooGripperError: If gripper not enabled or command fails
            BambooTimeoutError: If gripper command times out
        """
        if not self.enable_gripper:
            raise BambooGripperError("Gripper not enabled")

        command = {"command": "get_gripper_state"}

        if self.gripper_type == "franka":
            return self._send_panda_hand_command(command)
        else:
            return self._send_robotiq_command(command)


def main() -> None:
    """Simple test of the BambooFrankaClient."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Bamboo Franka Client")
    parser.add_argument("--port", default=5555, type=int, help="Port of bamboo control node")
    parser.add_argument("--ip", default="localhost", type=str, help="IP address of bamboo control node server")
    parser.add_argument("--samples", default=5, type=int, help="Number of state samples to fetch")
    parser.add_argument("--gripper-port", default=5559, type=int, help="ZMQ port of gripper server")
    parser.add_argument("--no-gripper", action="store_true", help="Disable gripper commands")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    print(f"Testing BambooFrankaClient at {args.ip}:{args.port}")

    try:
        with BambooFrankaClient(
            control_port=args.port,
            server_ip=args.ip,
            gripper_port=args.gripper_port,
            enable_gripper=not args.no_gripper,
        ) as client:
            print(f"\nFetching {args.samples} state samples:")
            print("-" * 60)

            for i in range(args.samples):
                result = client.get_joint_states()
                print(f"Sample {i + 1}:")
                print(f"  Joint positions: {[f'{q:.4f}' for q in result['qpos']]}")
                print(
                    f"  EE position: [{result['ee_pose'][0][3]:.4f}, "
                    f"{result['ee_pose'][1][3]:.4f}, {result['ee_pose'][2][3]:.4f}]"
                )
                print(f"  Gripper state: {result['gripper_state']}")

                if i < args.samples - 1:
                    time.sleep(0.2)  # Small delay between samples

        print("\nTest completed successfully!")

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error during test: {e}")


if __name__ == "__main__":
    main()
