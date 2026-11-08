"""Recording episodes as GIFs (needs a rendering backend: on Colab, the GPU runtime with MUJOCO_GL=egl)."""
import os

import jax
import numpy as np

from .evaluation import act_all_seeds


def save_gif(frames, path, seconds_per_frame, width=None):
    """Writes frames (H x W x 3 uint8 arrays) as a looping GIF with an explicit per-frame duration in milliseconds
    (GIF stores 1/100 s; values below 20 ms are not honored by browsers). width: optional downscaling."""
    from PIL import Image
    imgs = [Image.fromarray(np.asarray(f)[..., :3]) for f in frames]
    if width and imgs[0].width > width:
        h = int(round(imgs[0].height * width / imgs[0].width))
        imgs = [im.resize((width, h), Image.LANCZOS) for im in imgs]
    imgs = [im.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for im in imgs]
    ms = max(20, int(round(1000 * seconds_per_frame)))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=ms, loop=0, optimize=True, disposal=1)
    return path


def use_top_down_camera(env, size=480):
    """Maze tasks: a fixed camera above the maze center that shows the whole maze."""
    mz = env.unwrapped.maze
    inner = getattr(env.unwrapped, "ant_env", None) or getattr(env.unwrapped, "point_env")
    span = max(mz.map_width, mz.map_length) * mz.maze_size_scaling
    inner.model.vis.global_.offwidth = max(inner.model.vis.global_.offwidth, size)
    inner.model.vis.global_.offheight = max(inner.model.vis.global_.offheight, size)
    inner.mujoco_renderer.width = inner.mujoco_renderer.height = size
    inner.mujoco_renderer.default_cam_config = dict(
        lookat=np.array([0.0, 0.0, 0.0]), distance=span / (2 * np.tan(np.radians(22.5))),
        elevation=-90.0, azimuth=90.0)


def _torso_id(model):
    """Body id of the robot's torso (in AntMaze the maze blocks are bodies too, so it is not always 1)."""
    import mujoco
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    return i if i >= 0 else 1


def _extend_floor(model, half_size=2000.0):
    """Draw the floor much further out. MuJoCo draws a plane only up to its size (40 m for HalfCheetah, 20 m for
    Hopper), and a fast robot leaves that area within seconds. The checker texture is scaled by the same factor so
    the squares keep their size. Collisions with a plane do not depend on its size, so the physics is unchanged."""
    import mujoco
    for g in range(model.ngeom):
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE and 0 < model.geom_size[g][0] < half_size:
            f = half_size / model.geom_size[g][0]
            model.geom_size[g][:2] *= f
            if model.geom_matid[g] >= 0:
                model.mat_texrepeat[model.geom_matid[g]] *= f


def use_side_camera(env, width=640, height=360, distance=None, elevation=-12.0):
    """MuJoCo locomotion: a camera beside the robot that follows its torso (it would otherwise run out of view)."""
    inner = env.unwrapped
    inner.model.vis.global_.offwidth = max(inner.model.vis.global_.offwidth, width)
    inner.model.vis.global_.offheight = max(inner.model.vis.global_.offheight, height)
    inner.mujoco_renderer.width, inner.mujoco_renderer.height = width, height
    inner.mujoco_renderer.camera_id = None      # camera_id=-1 would reset the camera to "free" on every frame
    _extend_floor(inner.model)
    cheetah = "cheetah" in env.spec.id.lower()
    if distance is None:                       # frame the whole body: a bit more than its length
        distance = 3.6 if cheetah else 4.6
    # A free camera whose look-at point _render() moves to the torso before every frame. (MuJoCo's own tracking
    # camera follows with a lag, so a fast robot drifts to the edge of the picture.)
    inner._offrl_look_z = 0.35 if cheetah else 0.75      # height of the picture's center: mid-body
    inner.mujoco_renderer.default_cam_config = dict(
        distance=distance, elevation=elevation, azimuth=90.0, lookat=np.array([0.0, 0.0, inner._offrl_look_z]))


def _render(env):
    """env.render(), but for MuJoCo locomotion first center the side camera exactly on the torso."""
    u = env.unwrapped
    if hasattr(u, "_offrl_look_z"):
        viewer = u.mujoco_renderer._get_viewer(render_mode="rgb_array")
        x, y = u.data.xpos[_torso_id(u.model)][:2]
        viewer.cam.lookat[:] = [x, y, u._offrl_look_z]
    return env.render()


def use_chase_camera(env, size=480, distance=9.0, elevation=-50.0):
    """AntMaze: a camera above and behind the ant that follows it (shows the legs and the nearby walls)."""
    inner = env.unwrapped.ant_env
    inner.model.vis.global_.offwidth = max(inner.model.vis.global_.offwidth, size)
    inner.model.vis.global_.offheight = max(inner.model.vis.global_.offheight, size)
    inner.mujoco_renderer.width = inner.mujoco_renderer.height = size
    inner.mujoco_renderer.camera_id = None      # camera_id=-1 would reset the camera to "free" on every frame
    inner.mujoco_renderer.default_cam_config = dict(type=1, trackbodyid=_torso_id(inner.model), distance=distance,
                                                    elevation=elevation,
                                                    azimuth=90.0, lookat=np.zeros(3))


def _camera(task, env, top_down, camera_distance):
    if task.kind == "maze":
        if top_down:
            use_top_down_camera(env)
        elif task.is_ant:
            use_chase_camera(env, **({"distance": camera_distance} if camera_distance else {}))
    else:
        use_side_camera(env, distance=camera_distance)


def record_gif(task, states, normalizer, path, seed_index=0, seed=7000, reset_options=None,
               max_steps=None, every=None, top_down=True, camera_distance=None,
               width=640, height=360, size=480, fps=25, seconds=15.0, require_success=False, max_tries=20):
    """Runs one evaluation episode (mean action) and saves it as a GIF. Returns (path, status).

    MuJoCo: side camera centered on the torso, `width` x `height` pixels, played back in real time at `fps` frames
    per second, at most `seconds` of motion. Mazes: top-down camera (`size` pixels; for AntMaze `top_down=False`
    follows the ant). require_success=True (mazes): tries environment seeds seed, seed+1, ... (at most `max_tries`)
    and keeps the first episode that reaches the goal; the status says which seed was used."""
    pi = jax.tree.map(lambda x: x[seed_index:seed_index + 1], states["pi"])
    tries = max_tries if (require_success and task.kind == "maze") else 1
    for k in range(tries):
        env = task.make_env(render_mode="rgb_array")
        if task.kind == "maze":
            if top_down:
                use_top_down_camera(env, size=size)
            elif task.is_ant:
                use_chase_camera(env, size=size, **({"distance": camera_distance} if camera_distance else {}))
            n_every, duration, limit = every or 4, 0.05, max_steps or task.max_steps
        else:
            use_side_camera(env, width=width, height=height, distance=camera_distance)
            dt = env.unwrapped.dt
            n_every = every or max(1, int(round(1.0 / (fps * dt))))   # GIF frame delays below 0.02 s are not honored
            duration = dt * n_every
            limit = max_steps or min(task.max_steps, int(round(seconds / dt)))
        obs, _ = env.reset(seed=seed + k, options=reset_options)
        frames, ret, status, ok = [_render(env)], 0.0, "time limit", False
        for t in range(limit):
            a = np.asarray(act_all_seeds(pi, normalizer(task.state(obs)[None])))[0, 0]
            obs, r, term, trunc, info = env.step(a)
            ret += r
            if t % n_every == 0:
                frames.append(_render(env))
            if task.success(r, info):
                status, ok = f"success at step {t + 1}", True
                frames.append(_render(env)); break
            if term or trunc:
                status = f"ended at step {t + 1}" + (" (flipped)" if task.flipped(obs) else ""); break
        env.close()
        if ok or k == tries - 1:
            break
    if require_success and task.kind == "maze":
        status += f" (environment seed {seed + k}, attempt {k + 1})"
    save_gif(frames, path, duration)
    print(f"{status} | return {ret:.1f} | {len(frames)} frames, {1000 * duration:.0f} ms each -> {path} "
          f"({os.path.getsize(path) / 1e6:.1f} MB)")
    return path, status


def plot_filmstrip(task, states, normalizer, seed_index=0, seed=7000, n_frames=8, start=None, every=None,
                   top_down=False, camera_distance=None, title="", save_path=None):
    """One test episode as a row of snapshots (needs rendering, like record_gif).
    MuJoCo: n_frames consecutive snapshots from `start` (default: 200 steps in, i.e. the steady gait),
    `every` steps apart (default ~0.12 s), side camera following the torso; if the episode ends earlier
    (the robot fell), the last n_frames before the end instead.
    Maze: n_frames snapshots spread evenly over the whole episode (default camera: following the ant)."""
    import matplotlib.pyplot as plt
    env = task.make_env(render_mode="rgb_array")
    _camera(task, env, top_down, camera_distance)
    pi = jax.tree.map(lambda x: x[seed_index:seed_index + 1], states["pi"])
    obs, _ = env.reset(seed=seed)
    if task.kind == "maze":
        keep_every = 5
    else:
        every = every or max(1, int(round(0.12 / env.unwrapped.dt)))
        start = 200 if start is None else start
    frames, steps, status = [], [], "time limit"
    for t in range(task.max_steps):
        want = (t % keep_every == 0) if task.kind == "maze" else (t % every == 0)
        if want:
            frames.append(_render(env)); steps.append(t)
        if task.kind != "maze" and sum(st >= start for st in steps) >= n_frames:
            status = f"steps {start}-{t} of a test episode"; break
        a = np.asarray(act_all_seeds(pi, normalizer(task.state(obs)[None])))[0, 0]
        obs, r, term, trunc, info = env.step(a)
        if task.success(r, info):
            status = f"reached the goal at step {t + 1}"; frames.append(_render(env)); steps.append(t + 1); break
        if term or trunc:
            status = f"ended at step {t + 1}" + (" (flipped)" if task.flipped(obs) else ""); break
    env.close()
    if not frames:
        print("no frames recorded"); return
    if task.kind == "maze":                      # spread evenly over the episode, always including the last frame
        idx = np.unique(np.linspace(0, len(frames) - 1, min(n_frames, len(frames))).round().astype(int))
    elif sum(st >= start for st in steps) >= n_frames:
        idx = [i for i, st in enumerate(steps) if st >= start][:n_frames]
    else:                                        # ended (fell) before `start`: show its last moments
        idx = list(range(max(0, len(frames) - n_frames), len(frames)))
    frames, steps = [frames[i] for i in idx], [steps[i] for i in idx]
    u = env.unwrapped
    dt = getattr(u, "dt", None) or getattr(getattr(u, "ant_env", None) or getattr(u, "point_env", None), "dt", 0.0)
    if task.kind != "maze":                      # the side camera frames the robot in the middle: crop the margins
        h, w = frames[0].shape[:2]
        frames = [f[int(0.12 * h):int(0.92 * h), int(0.22 * w):int(0.78 * w)] for f in frames]
    aspect = frames[0].shape[0] / frames[0].shape[1]
    fig, axes = plt.subplots(1, len(frames), figsize=(2.4 * len(frames), 2.4 * aspect + 0.5))
    for ax, f, t in zip(np.atleast_1d(axes), frames, steps):
        ax.imshow(f); ax.axis("off"); ax.set_title(f"t = {t} ({t * dt:.2f} s)", fontsize=9)
    fig.suptitle((title + ": " if title else "") + status, y=1.02)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.show()
