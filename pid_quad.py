import functools as ft
from typing import NamedTuple

import ipdb
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

prev_Ja_B_sign = 0.0
n_sign_changes = 0


class Params(NamedTuple):
    m = 29.9  # mass
    g = 9.807  # gravity

    cf = 3.1582
    cd = 0.0079379

    # inertia
    Ixx = 0.001395
    Iyy = 0.001395
    Izz = 0.002173

    @property
    def J(self):
        return np.array([[self.Ixx, 0, 0], [0, self.Iyy, 0], [0, 0, self.Izz]])


class PIDParams(NamedTuple):
    kp: float = 400.0
    kv: float = 150.0
    kR: float = 10.0
    kw: float = 1.0


def get_R_W_B(phi, theta, psi) -> np.ndarray:
    yaw = np.array(
        [
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1],
        ]
    )
    pitch = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )
    roll = np.array(
        [
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)],
        ]
    )
    R_W_B = yaw @ pitch @ roll
    assert R_W_B.shape == (3, 3)
    return R_W_B


def get_cross(w: np.ndarray) -> np.ndarray:
    assert w.shape == (3,)
    wx, wy, wz = w
    return np.array(
        [
            [0, -wz, wy],
            [wz, 0, -wx],
            [-wy, wx, 0],
        ]
    )


def get_vee(M: np.ndarray) -> np.ndarray:
    assert M.shape == (3, 3)
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


def get_quad_xdot(state: np.ndarray, torque: np.ndarray, thrust: float | np.ndarray, params: Params = Params()):
    """
    state: (12,). [ px, py, pz | vx, vy, vz | ϕ, θ, ψ | p, q, r ]
    torque: (3,). Body frame.
    thrust: (). Body frame z-axis.
    """
    assert state.shape == (12,)
    assert torque.shape == (3,)
    assert isinstance(thrust, float) or thrust.shape == tuple()

    v_W = state[3:6]
    w_B = state[9:]
    phi, theta, psi = state[6], state[7], state[8]

    e3 = np.array([0, 0, 1])
    R_W_B = get_R_W_B(phi, theta, psi)
    z_W_B = R_W_B @ e3

    ma_W = -params.m * params.g * e3 + z_W_B * thrust
    Ja_B = -np.cross(w_B, params.J @ w_B) + torque

    # Sanity check: Ja_B should not be repeatedly changing sign. If it does, then the simulation is unstable.
    global prev_Ja_B_sign, n_sign_changes
    Ja_B_sign = np.sign(Ja_B[0])
    if Ja_B_sign != prev_Ja_B_sign:
        n_sign_changes += 1
        prev_Ja_B_sign = Ja_B_sign
        if n_sign_changes > 10:
            logger.error("Ja_B sign changes too frequently. Simulation is unstable.")
            raise ValueError("Ja_B sign changes too frequently. Simulation is unstable.")
    else:
        n_sign_changes = 0

    lin_acc_W = ma_W / params.m
    ang_acc_B = np.linalg.inv(params.J) @ Ja_B

    sphi, cphi = np.sin(phi), np.cos(phi)
    ctheta = np.cos(theta)
    ttheta = np.tan(theta)
    R_euler_B = np.array(
        [
            [1.0, sphi * ttheta, cphi * ttheta],
            [0.0, cphi, -sphi],
            [0.0, sphi / ctheta, cphi / ctheta],
        ]
    )
    deuler = R_euler_B @ w_B

    return np.concatenate([v_W, lin_acc_W, deuler, ang_acc_B])


def sim_rk4(deriv_fn, x0: np.ndarray, dt: float, n_steps: int):
    T_x = [x0]
    x = x0
    for kk in range(n_steps):
        k1 = deriv_fn(x)
        k2 = deriv_fn(x + 0.5 * dt * k1)
        k3 = deriv_fn(x + 0.5 * dt * k2)
        k4 = deriv_fn(x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        T_x.append(x)

        # Stop if x is too large.
        if np.any(np.abs(x) > 100.0):
            break

    T_x = np.stack(T_x, axis=0)
    T_t = np.arange(len(T_x)) * dt
    return T_t, T_x


def hover_xdot(state: np.ndarray, params: Params = Params()):
    torque = np.zeros(3)
    thrust = params.m * params.g
    return get_quad_xdot(state, torque, thrust, params)


def pid_xdot(state: np.ndarray, x_des: np.ndarray, pid_params: PIDParams = PIDParams(), params: Params = Params()):
    assert state.shape == x_des.shape == (12,)

    kp, kv, kR, kw = pid_params

    p_W = state[:3]
    v_W = state[3:6]
    phi, theta, psi = state[6], state[7], state[8]
    w_B = state[9:]

    p_W_des = x_des[:3]
    v_W_des = x_des[3:6]
    psi_des = x_des[8]
    w_des = x_des[9:]
    alpha_des = np.zeros(3)  # Desired angular acceleration

    # Linear errors.
    err_pos_W = p_W - p_W_des
    err_vel_W = v_W - v_W_des

    # Desired force in the world frame. (3,)
    e3 = np.array([0, 0, 1])
    f_W_des = -kp * err_pos_W - kv * err_vel_W + params.m * params.g * e3  # Assume desired acceleration is 0.
    assert f_W_des.shape == (3,)
    R_W_B = get_R_W_B(phi, theta, psi)
    f_W_z = np.dot(f_W_des, R_W_B @ e3)

    x_W_des_tilde = np.array([np.cos(psi_des), np.sin(psi_des), 0])  # 2D projection of the desired yaw.
    z_W_des = f_W_des / np.linalg.norm(f_W_des)
    y_W_des = np.cross(z_W_des, x_W_des_tilde)
    y_W_des = y_W_des / np.linalg.norm(y_W_des)
    x_W_des = np.cross(y_W_des, z_W_des)
    x_W_des = x_W_des / np.linalg.norm(x_W_des)
    R_W_des = np.stack([x_W_des, y_W_des, z_W_des], axis=1)
    assert R_W_des.shape == (3, 3)

    # Rotational errors.
    err_R = 0.5 * get_vee(R_W_des.T @ R_W_B - R_W_B.T @ R_W_des)
    err_w_B = w_B - R_W_B.T @ R_W_des @ w_des

    w_B_cross = get_cross(w_B)
    # tau_B_extra_terms = -params.J @ (w_B_cross @ R_W_B.T @ R_W_des @ w_des - R_W_B.T @ R_W_des @ alpha_des)
    tau_B_extra_terms = 0.0
    # logger.info("tau_B first two: {}".format(-kR * err_R - kw * err_w_B))
    tau_B = -kR * err_R - kw * err_w_B + np.cross(w_B, params.J @ w_B) + tau_B_extra_terms

    return get_quad_xdot(state, tau_B, f_W_z, params)


def plot_trajectory(T_t: np.ndarray, T_x_dict: dict[str, np.ndarray]):
    labels = [
        r"$p_x$",
        r"$p_y$",
        r"$p_z$",
        r"$v_x$",
        r"$v_y$",
        r"$v_z$",
        r"$\phi$",
        r"$\theta$",
        r"$\psi$",
        r"$p$",
        r"$q$",
        r"$r$",
    ]
    fig, axes = plt.subplots(4, 3, layout="constrained", figsize=(12, 8))
    for ii in range(12):
        ax: plt.Axes = axes[ii // 3, ii % 3]
        for label, T_x in T_x_dict.items():
            ax.plot(T_t, T_x[:, ii], label=label)
        ax.set_title(labels[ii])
        ax.set_xlabel("Time (t)")

    # Make each row have the same y-axis limits.
    for ii in range(4):
        ymin = min(ax.get_ylim()[0] for ax in axes[ii])
        ymax = max(ax.get_ylim()[1] for ax in axes[ii])
        for ax in axes[ii]:
            ax.set_ylim(ymin, ymax)

    axes[0, -1].legend()
    return fig


def main():
    params = Params()

    dt = 5e-4
    n_steps = 6_000

    x0 = np.zeros(12)
    x0[:3] = np.array([1.0, 0.0, 1.0])
    # x0[6:9] = np.array([0.1, 0.0, 0.0])

    x_des = np.zeros(12)
    T_t, T_x = sim_rk4(ft.partial(pid_xdot, x_des=x_des, params=params), x0, dt, n_steps)

    T_x_des = np.repeat(x_des[None, :], len(T_t), axis=0)

    # T_t, T_x = sim_rk4(ft.partial(hover_xdot, params=params), x0, dt, n_steps)
    fig = plot_trajectory(T_t, {"Desired": T_x_des, "Real": T_x})
    fig.savefig("pid_quad.pdf")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
