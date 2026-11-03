import threading

import msgpack
import numpy as np
import zmq

from bamboo import BambooFrankaClient

STATE = {
    "q": [0.0] * 7,
    "dq": [0.0] * 7,
    "tau_J": [0.0] * 7,
    "O_T_EE": np.eye(4).T.reshape(-1).tolist(),
    "time_sec": 1.0,
}


def _serve(responses: dict, count: int):
    """Answer ``count`` requests; return the context, socket, thread, port and the requests seen."""

    context = zmq.Context()
    server = context.socket(zmq.REP)
    port = server.bind_to_random_port("tcp://127.0.0.1")
    requests: list[dict] = []

    def serve() -> None:
        while len(requests) < count:
            request = msgpack.unpackb(server.recv(), raw=False)
            requests.append(request)
            response = responses.get(request["command"], {"success": True, "error": ""})
            server.send(msgpack.packb(response, use_bin_type=True))

    thread = threading.Thread(target=serve)
    thread.start()
    return context, server, thread, port, requests


def _finish(context, server, thread) -> None:
    thread.join(timeout=1.0)
    server.close()
    context.term()
    assert not thread.is_alive()


def test_guiding_protocol_and_increasing_sequence() -> None:
    status = {"success": True, "error": "", "stale": False, "phase": 0, "reason": 0, "level": 0.0}
    context, server, thread, port, requests = _serve(
        {
            "get_robot_state": {"success": True, "data": STATE},
            "guide_update": status,
            "guide_resume": status,
        },
        count=6,
    )
    client = BambooFrankaClient(server_ip="127.0.0.1", control_port=port, enable_gripper=False)
    assert client.start_guiding(watchdog_ms=200, max_joint_velocity=0.3, contact_point=[0, 0, 0.2])["success"]
    assert client.update_guiding()["success"]
    assert client.update_guiding(level=0.4, direction=[1.0, 0.0, 0.0])["success"]
    assert client.resume_guiding()["success"]
    client.close()  # stops guiding on the way out
    _finish(context, server, thread)

    commands = [request["command"] for request in requests]
    assert commands == ["get_robot_state", "start_guide", "guide_update", "guide_update", "guide_resume", "stop_guide"]
    start = requests[1]["data"]
    assert start == {"watchdog_ms": 200, "max_joint_velocity": 0.3, "contact_point": [0.0, 0.0, 0.2]}
    first, second = requests[2]["data"], requests[3]["data"]
    assert second["seq"] > first["seq"]
    assert second["level"] == 0.4 and second["direction"] == [1.0, 0.0, 0.0]
    assert "direction" not in first


def test_new_state_fields_are_read_only_when_present() -> None:
    newer = dict(STATE, O_F_ext_hat_K=[0, 0, -2.0, 0, 0, 0], tau_ext_hat_filtered=[0.1] * 7, loop_mode=3)
    newer.update(guide_phase=3, guide_reason=1, guide_level=0.25)
    context, server, thread, port, _ = _serve({"get_robot_state": {"success": True, "data": newer}}, count=2)
    client = BambooFrankaClient(server_ip="127.0.0.1", control_port=port, enable_gripper=False)
    states = client.get_joint_states()
    client.close()
    _finish(context, server, thread)
    assert states["ext_wrench"][2] == -2.0
    assert states["loop_mode"] == "guide"
    assert states["guide_phase"] == "hold" and states["guide_hold_reason"] == "watchdog"
    assert states["guide_level"] == 0.25

    context, server, thread, port, _ = _serve({"get_robot_state": {"success": True, "data": STATE}}, count=2)
    client = BambooFrankaClient(server_ip="127.0.0.1", control_port=port, enable_gripper=False)
    older = client.get_joint_states()
    client.close()
    _finish(context, server, thread)
    assert "ext_wrench" not in older and "guide_phase" not in older


def test_set_load_validates_before_sending() -> None:
    context, server, thread, port, requests = _serve({"get_robot_state": {"success": True, "data": STATE}}, count=2)
    client = BambooFrankaClient(server_ip="127.0.0.1", control_port=port, enable_gripper=False)
    try:
        client.set_load(-1.0, [0, 0, 0.05])
    except ValueError:
        pass
    else:
        raise AssertionError("a negative mass must be rejected locally")
    assert client.set_load(0.9, [0.0, 0.0, 0.06])["success"]
    client.close()
    _finish(context, server, thread)
    assert requests[-1] == {"command": "set_load", "data": {"mass": 0.9, "center_of_mass": [0.0, 0.0, 0.06]}}


def test_contact_limits_protocol() -> None:
    applied = {"enabled": True, "max_force_n": 12.0, "guard_force_n": 18.0}
    context, server, thread, port, requests = _serve(
        {
            "get_robot_state": {"success": True, "data": STATE},
            "set_contact_limits": {"success": True, "error": "", "limits": applied},
        },
        count=4,
    )
    client = BambooFrankaClient(server_ip="127.0.0.1", control_port=port, enable_gripper=False)
    for bad in ({"max_force_n": float("nan")}, {"guard_force_n": -1.0}, {"contact_point": [0.0, 0.1]}):
        try:
            client.set_contact_limits(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} must be rejected locally")
    response = client.set_contact_limits(max_force_n=12, contact_point=[0, 0, 0.2], guard_force_n=18, shadow=True)
    assert response["limits"] == applied
    assert client.clear_contact_limits()["success"]
    assert client.yield_streaming("gel at its limit")["success"]
    client.close()
    _finish(context, server, thread)
    commands = [request["command"] for request in requests]
    assert commands == ["get_robot_state", "set_contact_limits", "clear_contact_limits", "yield_stream"]
    sent = requests[1]["data"]
    assert sent["max_force_n"] == 12.0 and sent["contact_point"] == [0.0, 0.0, 0.2] and sent["shadow"] is True
    assert sent["guard_impulse_ns"] == 0.05 and sent["guard_on_invalid"] is True and sent["collision_force_n"] == 0.0
    assert requests[3]["data"] == {"reason": "gel at its limit"}


def test_contact_and_leash_state_is_read_only_when_present() -> None:
    newer = dict(STATE, guide_lead=0.012, guide_ratchet=True, contact_scale=0.5, contact_push=24.0, guard_trip=2)
    context, server, thread, port, _ = _serve({"get_robot_state": {"success": True, "data": newer}}, count=2)
    client = BambooFrankaClient(server_ip="127.0.0.1", control_port=port, enable_gripper=False)
    states = client.get_joint_states()
    client.close()
    _finish(context, server, thread)
    assert states["guide_lead"] == 0.012 and states["guide_ratchet"] is True
    assert states["contact_scale"] == 0.5 and states["contact_push"] == 24.0 and states["guard_trip"] == "hard"


def test_close_never_waits_on_a_request_the_server_did_not_take() -> None:
    import time

    import zmq as zmq_module

    from bamboo.client import BambooTimeoutError

    context, server, thread, port, _ = _serve({"get_robot_state": {"success": True, "data": STATE}}, count=1)
    client = BambooFrankaClient(server_ip="127.0.0.1", control_port=port, enable_gripper=False)
    _finish(context, server, thread)  # the server goes away after answering once
    assert client.control_socket.getsockopt(zmq_module.LINGER) == 0
    try:
        client._send_panda_hand_command({"command": "get_capabilities"}, timeout_ms=200)
    except BambooTimeoutError:
        pass
    closer = threading.Thread(target=client.close)
    started = time.monotonic()
    closer.start()
    closer.join(timeout=3.0)
    assert not closer.is_alive(), "close() hung on an unsent request"
    assert time.monotonic() - started < 1.0
