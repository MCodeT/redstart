import marimo

__generated_with = "0.13.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Redstart: A Lightweight Reusable Booster""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(src="public/images/redstart.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Project Redstart is an attempt to design the control systems of a reusable booster during landing.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In principle, it is similar to SpaceX's Falcon Heavy Booster.

    >The Falcon Heavy booster is the first stage of SpaceX's powerful Falcon Heavy rocket, which consists of three modified Falcon 9 boosters strapped together. These boosters provide the massive thrust needed to lift heavy payloads—like satellites or spacecraft—into orbit. After launch, the two side boosters separate and land back on Earth for reuse, while the center booster either lands on a droneship or is discarded in high-energy missions.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(
        mo.Html("""
    <iframe width="560" height="315" src="https://www.youtube.com/embed/RYUr-5PYA7s?si=EXPnjNVnqmJSsIjc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>""")
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Dependencies""")
    return


@app.cell(hide_code=True)
def _():
    import scipy
    import scipy.integrate as sci

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    from tqdm import tqdm

    # The use of autograd is optional in this project, but it may come in handy!
    import autograd
    import autograd.numpy as np
    import autograd.numpy.linalg as la
    from autograd import isinstance, tuple
    return FFMpegWriter, FuncAnimation, mpl, np, plt, scipy, tqdm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The Model

    The Redstart booster in model as a rigid tube of length $2 \ell$ and negligible diameter whose mass $M$ is uniformly spread along its length. It may be located in 2D space by the coordinates $(x, y)$ of its center of mass and the angle $\theta$ it makes with respect to the vertical (with the convention that $\theta > 0$ for a left tilt, i.e. the angle is measured counterclockwise)

    This booster has an orientable reactor at its base ; the force that it generates is of amplitude $f>0$ and the angle of the force with respect to the booster axis is $\phi$ (with a counterclockwise convention).

    We assume that the booster is subject to gravity, the reactor force and that the friction of the air is negligible.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(mo.image(src="public/images/geometry.svg"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Constants

    For the sake of simplicity (this is merely a toy model!) in the sequel we assume that: 

      - the total length $2 \ell$ of the booster is 2 meters,
      - its mass $M$ is 1 kg,
      - the gravity constant $g$ is 1 m/s^2.

    This set of values is not realistic, but will simplify our computations and do not impact the structure of the booster dynamics.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Helpers

    ### Rotation matrix

    $$ 
    \begin{bmatrix}
    \cos \alpha & - \sin \alpha \\
    \sin \alpha &  \cos \alpha  \\
    \end{bmatrix}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(np):
    def R(alpha):
        return np.array([
            [np.cos(alpha), -np.sin(alpha)], 
            [np.sin(alpha),  np.cos(alpha)]
        ])
    return (R,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Videos

    It will be very handy to make small videos to visualize the evolution of our booster!
    Here is an example of how such videos can be made with Matplotlib and displayed in marimo.
    """
    )
    return


@app.cell(hide_code=True)
def _(FFMpegWriter, FuncAnimation, mo, np, plt, tqdm):
    def make_video(output):
        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        num_frames = 100
        fps = 30 # Number of frames per second

        def animate(frame_index):    
            # Clear the canvas and redraw everything at each step
            plt.clf()
            plt.xlim(0, 2*np.pi)
            plt.ylim(-1.5, 1.5)
            plt.title(f"Sine Wave Animation - Frame {frame_index+1}/{num_frames}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)

            x = np.linspace(0, 2*np.pi, 100)
            phase = frame_index / 10
            y = np.sin(x + phase)
            plt.plot(x, y, "r-", lw=2, label=f"sin(x + {phase:.1f})")
            plt.legend()

            pbar.update(1)

        pbar = tqdm(total=num_frames, desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=num_frames)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")

    _filename = "wave_animation.mp4"
    make_video(_filename)
    mo.show_code(mo.video(src=_filename))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Getting Started""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Constants

    Define the Python constants `g`, `M` and `l` that correspond to the gravity constant, the mass and half-length of the booster.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    g = 1.0
    M = 1.0
    l = 1
    return M, g, l


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Forces

    Compute the force $(f_x, f_y) \in \mathbb{R}^2$ applied to the booster by the reactor.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    \begin{align*}
    f_x & = -f \sin (\theta + \phi) \\
    f_y & = +f \cos(\theta +\phi)
    \end{align*}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Center of Mass

    Give the ordinary differential equation that governs $(x, y)$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    \begin{align*}
    M \ddot{x} & = -f \sin (\theta + \phi) \\
    M \ddot{y} & = +f \cos(\theta +\phi) - Mg
    \end{align*}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Moment of inertia

    Compute the moment of inertia $J$ of the booster and define the corresponding Python variable `J`.
    """
    )
    return


@app.cell
def _(M, l):
    J = M * l * l / 3
    J
    return (J,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Tilt

    Give the ordinary differential equation that governs the tilt angle $\theta$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    J \ddot{\theta} = - \ell (\sin \phi)  f
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Simulation

    Define a function `redstart_solve` that, given as input parameters: 

      - `t_span`: a pair of initial time `t_0` and final time `t_f`,
      - `y0`: the value of the state `[x, dx, y, dy, theta, dtheta]` at `t_0`,
      - `f_phi`: a function that given the current time `t` and current state value `y`
         returns the values of the inputs `f` and `phi` in an array.

    returns:

      - `sol`: a function that given a time `t` returns the value of the state `[x, dx, y, dy, theta, dtheta]` at time `t` (and that also accepts 1d-arrays of times for multiple state evaluations).

    A typical usage would be:

    ```python
    def free_fall_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, y):
            return np.array([0.0, 0.0]) # input [f, phi]
        sol = redstart_solve(t_span, y0, f_phi)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Free Fall")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    free_fall_example()
    ```

    Test this typical example with your function `redstart_solve` and check that its graphical output makes sense.
    """
    )
    return


@app.cell(hide_code=True)
def _(J, M, g, l, np, scipy):
    def redstart_solve(t_span, y0, f_phi):
        def fun(t, state):
            x, dx, y, dy, theta, dtheta = state
            f, phi = f_phi(t, state)
            d2x = (-f * np.sin(theta + phi)) / M
            d2y = (+ f * np.cos(theta + phi)) / M - g
            d2theta = (- l * np.sin(phi)) * f / J
            return np.array([dx, d2x, dy, d2y, dtheta, d2theta])
        r = scipy.integrate.solve_ivp(fun, t_span, y0, dense_output=True)
        return r.sol
    return (redstart_solve,)


@app.cell(hide_code=True)
def _(l, np, plt, redstart_solve):
    def free_fall_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, y):
            return np.array([0.0, 0.0]) # input [f, phi]
        sol = redstart_solve(t_span, y0, f_phi)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Free Fall")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    free_fall_example()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controlled Landing

    Assume that $x$, $\dot{x}$, $\theta$ and $\dot{\theta}$ are null at $t=0$. For $y(0)= 10$ and $\dot{y}(0) = - 2*\ell$,  can you find a time-varying force $f(t)$ which, when applied in the booster axis ($\theta=0$), yields $y(5)=\ell$ and $\dot{y}(5)=0$?

    Simulate the corresponding scenario to check that your solution works as expected.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    % y(t)
    y(t)
    = \frac{2(5-\ell)}{125}\,t^3
      + \frac{3\ell-10}{25}\,t^2
      - 2\,t
      + 10
    $$

    $$
    % f(t)
    f(t)
    = M\!\Bigl[
        \frac{12(5-\ell)}{125}\,t
        + \frac{6\ell-20}{25}
        + g
      \Bigr].
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(M, g, l, np, plt, redstart_solve):

    def smooth_landing_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi_smooth_landing(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi=f_phi_smooth_landing)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Controlled Landing")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    smooth_landing_example()
    return


@app.cell
def _(M, g, l, np, plt, redstart_solve):
    def smooth_landing_example_force():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi_smooth_landing(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi=f_phi_smooth_landing)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Controlled Landing")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    smooth_landing_example_force()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Drawing

    Create a function that draws the body of the booster, the flame of its reactor as well as its target landing zone on the ground (of coordinates $(0, 0)$).

    The drawing can be very simple (a rectangle for the body and another one of a different color for the flame will do perfectly!).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(mo.image("public/images/booster_drawing.png"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Make sure that the orientation of the flame is correct and that its length is proportional to the force $f$ with the length equal to $\ell$ when $f=Mg$.

    The function shall accept the parameters `x`, `y`, `theta`, `f` and `phi`.
    """
    )
    return


@app.cell(hide_code=True)
def _(M, R, g, l, mo, mpl, np, plt):
    def draw_booster(x=0, y=l, theta=0.0, f=0.0, phi=0.0, axes=None, **options):
        L = 2 * l
        if axes is None:
            _fig, axes = plt.subplots()

        axes.set_facecolor('#F0F9FF') 

        ground = np.array([[-2*l, 0], [2*l, 0], [2*l, -l], [-2*l, -l], [-2*l, 0]]).T
        axes.fill(ground[0], ground[1], color="#E3A857", **options)

        b = np.array([
            [l/10, -l], 
            [l/10, l], 
            [0, l+l/10], 
            [-l/10, l], 
            [-l/10, -l], 
            [l/10, -l]
        ]).T
        b = R(theta) @ b
        axes.fill(b[0]+x, b[1]+y, color="black", **options)

        ratio = l / (M*g) # when f= +MG, the flame length is l 

        flame = np.array([
            [l/10, 0], 
            [l/10, - ratio * f], 
            [-l/10, - ratio * f], 
            [-l/10, 0], 
            [l/10, 0]
        ]).T
        flame = R(theta+phi) @ flame
        axes.fill(
            flame[0] + x + l * np.sin(theta), 
            flame[1] + y - l * np.cos(theta), 
            color="#FF4500", 
            **options
        )

        return axes

    _axes = draw_booster(x=0.0, y=20*l, theta=np.pi/8, f=M*g, phi=np.pi/8)
    _fig = _axes.figure
    _axes.set_xlim(-4*l, 4*l)
    _axes.set_ylim(-2*l, 24*l)
    _axes.set_aspect("equal")
    _axes.grid(True)
    _MaxNLocator = mpl.ticker.MaxNLocator
    _axes.xaxis.set_major_locator(_MaxNLocator(integer=True))
    _axes.yaxis.set_major_locator(_MaxNLocator(integer=True))
    _axes.set_axisbelow(True)
    mo.center(_fig)
    return (draw_booster,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Visualisation

    Produce a video of the booster for 5 seconds when

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=0$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=\pi/8$

      - the parameters are those of the controlled landing studied above.

    As an intermediary step, you can begin the with production of image snapshots of the booster location (every 1 sec).
    """
    )
    return


@app.cell(hide_code=True)
def _(draw_booster, l, mo, np, plt, redstart_solve):
    def sim_1():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([0.0, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_1()
    return


@app.cell(hide_code=True)
def _(M, draw_booster, g, l, mo, np, plt, redstart_solve):
    def sim_2():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_2()
    return


@app.cell(hide_code=True)
def _(M, draw_booster, g, l, mo, np, plt, redstart_solve):
    def sim_3():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, np.pi/8])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_3()
    return


@app.cell(hide_code=True)
def _(M, draw_booster, g, l, mo, np, plt, redstart_solve):
    def sim_4():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_4()
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    draw_booster,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_1():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([0.0, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_1.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_1())
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    M,
    draw_booster,
    g,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_2():
        L = 2*l

        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_2.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_2())
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    M,
    draw_booster,
    g,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_3():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, np.pi/8])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_3.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_3())
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    M,
    draw_booster,
    g,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_4():
        L = 2*l
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_4.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_4())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Linearized Dynamics""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Equilibria

    We assume that $|\theta| < \pi/2$, $|\phi| < \pi/2$ and that $f > 0$. What are the possible equilibria of the system for constant inputs $f$ and $\phi$ and what are the corresponding values of these inputs?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Pour qu’un état soit à l’équilibre, les variables doivent rester constantes ; toutes les dérivées sont donc nulles :  
     \(
        \dot{x} = \ddot{x} = 0,\quad \dot{y} = \ddot{y} = 0,\quad \dot{\theta} = \ddot{\theta} = 0
     \)

    On impose les contraintes physiques :  
    \( |\theta| < \dfrac{\pi}{2} \), \( |\phi| < \dfrac{\pi}{2} \) et \( f>0 \).

    À l’équilibre, toutes les accélérations sont nulles; on résout donc :


    ### 1. Accélération horizontale
    \(
    \ddot{x}= -\frac{f}{M}\,\sin(\theta+\phi)=0
    \quad\Longrightarrow\quad
    \sin(\theta+\phi)=0
    \quad\Longrightarrow\quad
    \theta+\phi = n\pi,\; n\in\mathbb{Z}.
    \)

    Sous les contraintes d’angle (\(|\theta|,|\phi|<\pi/2\)), la seule solution admissible est  
    \(
    \boxed{\theta+\phi = 0 \;\Longrightarrow\; \phi = -\theta}.
    \)


    ### 2. Accélération verticale
    \(
    \ddot{y}= \frac{f}{M}\cos(\theta+\phi)-g = 0
    \quad\Longrightarrow\quad
    \frac{f}{M}\cos(\theta+\phi) = g.
    \)

    En utilisant \(\theta+\phi=0\) donc \(\cos(\theta+\phi)=1\) :
    \(
    \boxed{f = M\,g}.
    \)


    ### 3. Accélération angulaire
    \(
    \ddot{\theta}= -\frac{\ell f}{J}\,\sin(\phi)=0
    \quad\Longrightarrow\quad
    \sin(\phi)=0
    \quad\Longrightarrow\quad
    \phi = 0
    \quad\Longrightarrow\quad
    \boxed{\theta = 0} \;\;(\text{car } \phi = -\theta).
    \)


    ### Conditions d’équilibre

    Le système est donc à l’équilibre lorsque :

    * \(\boxed{\theta = 0}\)
    * \(\boxed{\phi = 0}\)
    * \(\boxed{f = Mg}\)


     *Application numérique* :  
     pour \(M = 1\,\text{kg}\) et \(g = 1\,\text{m/s}^2\)  
     \(\Rightarrow f = 1\,\text{N}\).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Linearized Model

    Introduce the error variables $\Delta x$, $\Delta y$, $\Delta \theta$, and $\Delta f$ and $\Delta \phi$ of the state and input values with respect to the generic equilibrium configuration.
    What are the linear ordinary differential equations that govern (approximately) these variables in a neighbourhood of the equilibrium?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Définissons les petites déviations autour de l’équilibre précédent :

    \[
    \Delta x = x - x^*,\quad
    \Delta y = y - y^*,\quad
    \Delta \theta = \theta - 0,\quad
    \Delta f = f - M\,g,\quad
    \Delta \phi = \phi - 0
    \]

    Nous notons également \(\Delta \dot x\), \(\Delta \dot y\), \(\Delta \dot \theta\) les déviations de vitesses.

    En linéarisant l’EDO non linéaire au premier ordre (en négligeant les produits de petits termes) :

    \[
    \Delta \ddot x 
    = -\frac{1}{M}\,(M\,g + \Delta f)\,\sin(\Delta\theta + \Delta\phi)
    \;\approx\;
    -\frac{M\,g}{M}\,(\Delta\theta + \Delta\phi)
    \;=\;
    -g\,(\Delta\theta + \Delta\phi),
    \]

    \[
    \Delta \ddot y 
    = \frac{M\,g + \Delta f}{M}\,\cos(\Delta\theta + \Delta\phi)\;-\;g
    \;\approx\;
    \frac{\Delta f}{M},
    \]

    \[
    \Delta \ddot \theta
    = -\frac{l\,(M\,g + \Delta f)}{J}\,\sin(\Delta\phi)
    \;\approx\;
    -\frac{l\,M\,g}{J}\,\Delta\phi.
    \]

    alors:

    \[
    \begin{aligned}
    \ddot{\Delta x} &= -g(\Delta \theta + \Delta \phi) \\
    \ddot{\Delta y} &= \frac{1}{M} \Delta f \\
    \ddot{\Delta \theta} &= -\frac{Mg\ell}{J} \Delta \phi
    \end{aligned}
    \]
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Standard Form

    What are the matrices $A$ and $B$ associated to this linear model in standard form?
    Define the corresponding NumPy arrays `A` and `B`.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    On écrit le système linéarisé sous la forme d’état classique :

    \[
    \dot{X} = A X + B U
    \quad \text{avec} \quad
    X =
    \begin{bmatrix}
    \Delta x \\
    \Delta \dot{x} \\
    \Delta y \\
    \Delta \dot{y} \\
    \Delta \theta \\
    \Delta \dot{\theta}
    \end{bmatrix},
    \quad
    U =
    \begin{bmatrix}
    \Delta f \\
    \Delta \phi
    \end{bmatrix}
    \]

    Les matrices \( A \) et \( B \) sont données par :

    \[
    A =
    \begin{bmatrix}
    0 & 1 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & -g & 0 \\
    0 & 0 & 0 & 1 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0 & 0 & 0 \\
    \end{bmatrix}
    \quad , \quad
    B =
    \begin{bmatrix}
    0 & 0 \\
    0 & -g \\
    0 & 0 \\
    \frac{1}{M} & 0 \\
    0 & 0 \\
    0 & -\frac{Mg\ell}{J} \\
    \end{bmatrix}
    \]
    """
    )
    return


@app.cell
def _(J, M, g, l, np):
    A = np.array([
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, -g, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ])

    B = np.array([
        [0, 0],
        [0, -g],
        [0, 0],
        [1/M, 0],
        [0, 0],
        [0, -M * g * l / J],
    ])

    return A, B


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Stability

    Is the generic equilibrium asymptotically stable?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Calcul des valeurs propres de \(A\) :""")
    return


@app.cell
def _(A, np):
    valeurs_propres = np.linalg.eigvals(A)

    print("Valeurs propres de la matrice A :")
    print(valeurs_propres)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    La stabilité du système linéaire est déterminée par les *valeurs propres* de la matrice \(A\), c’est-à-dire les racines du polynôme caractéristique \(\det(A - \lambda I) = 0\).


    On calcule les valeurs propres, on trouve :

    \[
    \lambda \in \{ 0, 0, 0, 0, 0, 0 \}
    \]

    Les valeurs propres n'étant pas à parties réelles strictement négatives, le système **n’est pas asymptotiquement stable**.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controllability

    Is the linearized model controllable?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    On analyse la matrice de contrôlabilité $\mathcal{C}$:

    Si \( \text{rang}(\mathcal{C}) = 6 \), alors le système est complètement contrôlable.

    \[
    \mathcal{C} = [B, AB, A^2B, A^3B, A^4B, A^5B] \in \mathbb{R}^{6 \times 12}
    \]


    On peut raisonner sur  \( \Delta f \) et \( \Delta \phi \) comme suit :

    - **\( \Delta f \)** : a un effet direst sur la dynamique verticale \( \Delta \ddot{y} \).

    - **\( \Delta \phi \)** : a un effet à la fois sur la dynamique angulaire \( \Delta \ddot{\theta} \) et sur la position horizontale \( \Delta \ddot{x} \) via son influence sur \( \Delta \theta \).



      En effet, même si \( \Delta \phi \) n'agit pas directement sur \( \Delta x \), elle influence l’angle \( \theta \), qui à son tour agit sur le mouvement horizontal dans les itérations suivantes via la matrice \( A \).

    Tous les états deviennent accessibles via l’action des entrées, ce qui signifie que le modèle linéarisé est **contrôlable**.
    """
    )
    return


@app.cell
def _(A):
    A.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""On fait le calcul du rang directement avec Python pour verifier :""")
    return


@app.cell
def _(A, B, np):
    from scipy.linalg import block_diag

    # Matrice de contrôlabilité
    n = 6
    CB = B
    for i in range(1, n):
        CB = np.hstack((CB, np.linalg.matrix_power(A, i) @ B))

    # Rang
    rang_CB = np.linalg.matrix_rank(CB)
    print("Rang de la matrice de contrôlabilité =", rang_CB)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Lateral Dynamics

    We limit our interest in the lateral position $x$, the tilt $\theta$ and their derivatives (we are for the moment fine with letting $y$ and $\dot{y}$ be uncontrolled). We also set $f = M g$ and control the system only with $\phi$.

    What are the new (reduced) matrices $A$ and $B$ for this reduced system?
    Check the controllability of this new system.
    """
    )
    return


@app.cell
def _(g, l):
    import numpy as nn
    from numpy.linalg import matrix_rank

    # Définition des matrices réduites
    A_lat = nn.array([
        [0, 1, 0, 0],
        [0, 0, -g, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])

    B_lat = nn.array([
        [0],
        [-g],
        [0],
        [-3 * g / l]
    ])

    # Matrice de contrôlabilité
    C_lat = nn.hstack([B_lat, A_lat @ B_lat, A_lat @ A_lat @ B_lat, A_lat @ A_lat @ A_lat @ B_lat])
    rank_C_lat = matrix_rank(C_lat)
    print("Rang de la matrice de contrôlabilité latérale :", rank_C_lat)
    return A_lat, B_lat


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    On pose les hypothéses suivantes:

    - On néglige la dynamique verticale (\(y, \dot{y}\)) 
    - On fixe la poussée à \(f = Mg\)
    - Le seul signal de commande est \(\phi\),l’orientation du moteur.
    - Le vecteur d’état s'exprime donc par :

    \[
    \Delta x_{\text{lat}} = 
    \begin{bmatrix}
    \Delta x \\
    \Delta \dot{x} \\
    \Delta \theta \\
    \Delta \dot{\theta}
    \end{bmatrix}
    \in \mathbb{R}^4
    \]

    et l’entrée :

    \[
    \Delta u_{\text{lat}} = \Delta \phi \in \mathbb{R}
    \]


    La dynamique réduite s’écrit :

    \[
    \dot{\Delta x}_{\text{lat}} = A_{\text{lat}} \Delta x_{\text{lat}} + B_{\text{lat}} \Delta \phi
    \]

    avec :

    \[
    A_{\text{lat}} = \begin{bmatrix}
    0 & 1 & 0 & 0 \\
    0 & 0 & -g & 0 \\
    0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0
    \end{bmatrix}, \quad
    B_{\text{lat}} = \begin{bmatrix}
    0 \\
    - g \\
    0 \\
    -\frac{Mg\ell}{J} \\
    \end{bmatrix}
    \]


    Ensuite, on vérifie la contrôlabilité avec la matrice suivante :

    \[
    \mathcal{C}_{\text{lat}} = \begin{bmatrix}
    B_{\text{lat}} & A_{\text{lat}} B_{\text{lat}} & A_{\text{lat}}^2 B_{\text{lat}} & A_{\text{lat}}^3 B_{\text{lat}}
    \end{bmatrix}
    \in \mathbb{R}^{4 \times 4}
    \]

    Le code python nous donne un rang égale à *4* 


    Conclusion: Le système latéral est complètement contrôlable, ce qui permet de : Concevoir une commande sur \(\phi\)  seule et stabiliser le booster latéralement (angle et position).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Linear Model in Free Fall

    Make graphs of $y(t)$ and $\theta(t)$ for the linearized model when $\phi(t)=0$,
    $x(0)=0$, $\dot{x}(0)=0$, $\theta(0) = 45 / 180  \times \pi$  and $\dot{\theta}(0) =0$. What do you see? How do you explain it?
    """
    )
    return


@app.cell(hide_code=True)
def _(g, np, plt):

    theta0 = np.deg2rad(45)
    y0 = 0.0                 
    y_dot0 = 0.0              
    theta_dot0 = 0.0  

    t = np.linspace(0, 5, 500)

    y = y0 + y_dot0 * t - 0.5 * g * t**2
    theta = theta0 + theta_dot0 * t

    # Plot y(t)
    plt.figure()
    plt.plot(t, y)
    plt.title("Chute libre linéarisée: $x(t)$")
    plt.xlabel("temps $t$")
    plt.ylabel("$y(t)$")
    plt.grid(True)
    plt.show()

    # Plot θ(t)
    plt.figure()
    plt.plot(t, theta)
    plt.title("Chute libre linéarisée : $\\theta(t)$")
    plt.xlabel("temps $t$")
    plt.ylabel("$\\theta(t)$")
    yticks = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    yticklabels = ['0', 'π/4', 'π/2', '3π/4', 'π']
    plt.yticks(yticks, yticklabels)
    plt.grid(True)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### What we see :

    - *\(y(t) :\)* On obtient une trajectoire en chute libre :  
      \(
        y(t) \;=\; y(0) \;+\; \dot y(0)\,t \;-\;\tfrac12\,g\,t^2
        \;=\;-\,\tfrac12\,t^2
      \)
      avec \(g=1\), \(y(0)=0\) et \(\dot y(0)=0\).

    - *\(\theta(t) :\)* L’angle reste constant à sa valeur initiale :  
      \(
        \theta(t) \;=\;\theta(0)
        \;=\;\tfrac{\pi}{4}
      \)
      car la vitesse angulaire initiale est nulle et il n’y a pas de couple de redressement.
    ### why :

    1) Équation de translation verticale :  
     -  Dans le modèle linéarisé autour de l’équilibre de vol stationnaire, on a :
       \(
         \Delta\ddot y \;=\;\tfrac{1}{M}\,\Delta f\,.
       \)  
       Pour la chute libre, on pose \(f(t)=0\), donc  
       \(
         \Delta f = f - Mg = -\,Mg
         \quad\Longrightarrow\quad
         \Delta\ddot y = -\,g.
       \)  
       D’où la loi \(y(t)=y(0)+\dot y(0)\,t-\tfrac12\,g\,t^2\).

    2) Équation de rotation :  
    - Toujours en linéarisant :  
       \(
         \Delta\ddot\theta 
         \;=\; -\,\frac{M\,g\,\ell}{J}\,\Delta\varphi\,.
       \)  
       Ici \(\varphi(t)=0\) ⇒ \(\Delta\varphi=0\) ⇒  
       \(\Delta\ddot\theta=0\).  
       Avec \(\dot\theta(0)=0\), on déduit :
       \(
         \theta(t)=\theta(0)
         \;=\;\tfrac{\pi}{4}\,,
       \) 
       d’où l’angle reste constant.  

    Donc la chute libre entraîne la parabole en \(y(t)\), tandis que l’absence de commande angulaire (\(\varphi=0\)) fige la rotation.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Manually Tuned Controller

    Try to find the two missing coefficients of the matrix 

    $$
    K =
    \begin{bmatrix}
    0 & 0 & ? & ?
    \end{bmatrix}
    \in \mathbb{R}^{4\times 1}
    $$ 

    such that the control law 

    $$
    \Delta \phi(t)
    = 
    - K \cdot
    \begin{bmatrix}
    \Delta x(t) \\
    \Delta \dot{x}(t) \\
    \Delta \theta(t) \\
    \Delta \dot{\theta}(t)
    \end{bmatrix} \in \mathbb{R}
    $$

    manages  when
    $\Delta x(0)=0$, $\Delta \dot{x}(0)=0$, $\Delta \theta(0) = 45 / 180  \times \pi$  and $\Delta \dot{\theta}(0) =0$ to: 

      - make $\Delta \theta(t) \to 0$ in approximately $20$ sec (or less),
      - $|\Delta \theta(t)| < \pi/2$ and $|\Delta \phi(t)| < \pi/2$ at all times,
      - (but we don't care about a possible drift of $\Delta x(t)$).

    Explain your thought process, show your iterations!

    Is your closed-loop model asymptotically stable?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    On veut un controleur d'état de la forme :

    $$
    \Delta\phi(t) = -K \cdot 
    \begin{bmatrix}
    \Delta x(t) \\
    \Delta \dot{x}(t) \\
    \Delta \theta(t) \\
    \Delta \dot{\theta}(t)
    \end{bmatrix}, \quad
    \text{avec} \quad K = 
    \begin{bmatrix}
    0 & 0 & k_3 & k_4
    \end{bmatrix}
    $$

    Seules les variables $\theta$ et $\dot{\theta}$ sont utilisées pour le retour d'état.



    On étudie un **sous-système** constitué de $(\theta, \dot{\theta})$, 
    On a un systeme inspire du pendule inverse :

    $$
    \Delta \ddot{\theta}(t) = a \cdot \Delta \theta(t) + b \cdot \Delta \phi(t)
    $$

    Puisque :

    $$
    \Delta \phi(t) = -k_3 \cdot \Delta \theta(t) - k_4 \cdot \Delta \dot{\theta}(t)
    $$

    On obtient finalement :

    $$
    \Delta \ddot{\theta} = (a - b k_3) \cdot \Delta \theta - b k_4 \cdot \Delta \dot{\theta}
    $$

    On reconnaît l’équation d’un système du **deuxième ordre** :

    $$
    \Delta \ddot{\theta} + 2 \zeta \omega_n \cdot \Delta \dot{\theta} + \omega_n^2 \cdot \Delta \theta = 0
    $$

    Avec :

    - $\zeta$ : facteur d’amortissement

    - $\omega_n$ : pulsation naturelle

    Par identification :

    $$
    \begin{aligned}
    2 \zeta \omega_n &= b k_4 \quad \Rightarrow \quad k_4 = \frac{2 \zeta \omega_n}{b} \\
    \omega_n^2 &= b k_3 - a \quad \Rightarrow \quad k_3 = \frac{\omega_n^2 + a}{b}
    \end{aligned}
    $$

    On choisit les paramètres

    Par exemple :
    - $\zeta = 0.7$ (bon compromis entre rapidité et amortissement)
    - $\omega_n = 0.5$ rad/s (réponse lente mais stable)

    Alors :

    $$
    \begin{aligned}
    k_4 &= \frac{2 \cdot 0.7 \cdot 0.5}{b} = \frac{0.7}{b} \\
    k_3 &= \frac{0.5^2 + a}{b} = \frac{0.25 + a}{b}
    \end{aligned}
    $$

    On va estimer les constantes **a** et **b** pour avoir un systeme stable et correct.


    Par simulation

    On teste plusieurs valeurs, on trouve par exemple : $k_3 = 0.25$, $k_4 = 0.6$ sont les bons coefficients,

    On fait la simulation avec une inclinaison initiale : $\theta(0) = \frac{\pi}{4}$

    Pour objectifs :
       - $\theta(t) \to 0$ en moins de 20 s
       - $|\theta(t)| < \frac{\pi}{2}$
       - $|\phi(t)| < \frac{\pi}{2}$

    Et on ajuste :
       - Si $\theta$ est trop lent : **augmenter $k_3$**
       - Si $\theta$ oscille trop : **augmenter $k_4$**
       - Si $\phi$ devient trop grand : **réduire les deux gains**


    **Un bon réglage de $(k_3, k_4)$ permet de ramener rapidement l’angle $\theta$ à zéro, tout en respectant les contraintes physiques du système.**
    """
    )
    return


@app.cell
def _(np, plt):
    def _():
        from scipy.integrate import solve_ivp

        # Liste de couples (K3, K4) à tester
        gain_list = [
            (0.04,   0.4),
            (0.0625, 0.5),
            (0.2,    0.6),
            (0.25,   0.6)
        ]

        # Fonction de simulation pour (K3, K4)
        def simulate_pair(K3, K4, t_final=20, num_pts=1000, settle_tol=1e-2):
            # Conditions initiales
            delta_theta0 = np.pi/4
            theta_dot0 = 0.0

            # Dynamique
            def dynamics(t, y):
                dth, dth_dot = y
                # loi de commande
                delta_phi = -K3 * dth - K4 * dth_dot
                # saturation
                delta_phi = np.clip(delta_phi, -np.pi/2, np.pi/2)
                return [dth_dot, delta_phi]

            # Intégration
            t_eval = np.linspace(0, t_final, num_pts)
            sol = solve_ivp(dynamics, (0, t_final), [delta_theta0, theta_dot0], t_eval=t_eval)
            delta_theta = sol.y[0]
            phi_cmd = np.clip(-K3 * delta_theta - K4 * sol.y[1], -np.pi/2, np.pi/2)

            # Calcul du temps de stabilisation
            above = np.abs(delta_theta) > settle_tol
            if np.any(above):
                last_idx = np.max(np.where(above))
                Ts = sol.t[last_idx + 1] if last_idx < len(sol.t) - 1 else np.nan
            else:
                Ts = 0.0

            return sol.t, delta_theta, phi_cmd, Ts

        # Création du graphique
        fig, (ax_dt, ax_phi) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

        for K3, K4 in gain_list:
            t_vals, dtheta_vals, phi_vals, Ts = simulate_pair(K3, K4)
            label = f"K3={K3:.3f}, K4={K4:.3f}, Ts={Ts:.1f}s" if not np.isnan(Ts) else f"K3={K3:.3f}, K4={K4:.3f}, Ts=nan"
            ax_dt.plot(t_vals, dtheta_vals, label=label)
            ax_phi.plot(t_vals, phi_vals, label=label)

        # Lignes de borne ±π/2
        for ax in (ax_dt, ax_phi):
            ax.axhline( np.pi/2, color='gray', ls='--')
            ax.axhline(-np.pi/2, color='gray', ls='--')
            ax.grid(True)

        # Zoom vertical
        ax_dt.set_ylim(-0.5, 0.85)
        ax_phi.set_ylim(-0.1, 0.1)

        # Labels et titres
        ax_dt.set_ylabel("Δθ(t) (rad)")
        ax_dt.set_title("Comparaison de Δθ(t) pour différents gains")
        ax_phi.set_ylabel("Δφ(t) (rad)")
        ax_phi.set_xlabel("Temps (s)")
        ax_phi.set_title("Commande saturée Δφ(t)")

        # Légende
        ax_dt.legend(fontsize='small', loc='upper right')
        ax_phi.legend(fontsize='small', loc='upper right')

        plt.tight_layout()
        return plt.show()


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controller Tuned with Pole Assignment

    Using pole assignement, find a matrix

    $$
    K_{pp} =
    \begin{bmatrix}
    ? & ? & ? & ?
    \end{bmatrix}
    \in \mathbb{R}^{4\times 1}
    $$ 

    such that the control law 

    $$
    \Delta \phi(t)
    = 
    - K_{pp} \cdot
    \begin{bmatrix}
    \Delta x(t) \\
    \Delta \dot{x}(t) \\
    \Delta \theta(t) \\
    \Delta \dot{\theta}(t)
    \end{bmatrix} \in \mathbb{R}
    $$

    satisfies the conditions defined for the manually tuned controller and additionally:

      - result in an asymptotically stable closed-loop dynamics,

      - make $\Delta x(t) \to 0$ in approximately $20$ sec (or less).

    Explain how you find the proper design parameters!
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    ###REP:
    Pour concevoir la matrice de rétroaction \( K_{pp} \) pour les dynamiques latérales, nous utilisons la méthode de placement de pôles appliquée au système réduit :

    \[
    A_{\text{cl}} = A_{\text{lat}} - B_{\text{lat}} K_{pp}
    \]

    On a pour objectif d’obtenir :

    - une **stabilité asymptotique** du système en boucle fermée,
    - une **convergence rapide** de l’erreur de position latérale \( \Delta x(t) \), idéalement en moins de 20 secondes.

    Pour le choix des pôles:

    Après plusieurs tests, nous avons choisi d'accélérer la dynamique en plaçant les pôles plus à gauche :

    \[
    \texttt{desired\_poles} = [-2,\ -2.8,\ -3.1,\ -3.7]
    \]

    Ces valeurs garantissent non seulement la stabilité mais aussi une convergence beaucoup plus rapide que l’objectif initial.

    La matrice de gains obtenue à l’aide de `place_poles` est :

    \[
    K_{pp} = [k_1,\ k_2,\ k_3,\ k_4]
    \]

    Elle définit la loi de commande :

    \[
    \Delta \phi(t) = -K_{pp} \cdot \mathbf{x}_{\text{lat}}(t)
    \]

    où \( \mathbf{x}_{\text{lat}}(t) = [\Delta x,\ \Delta \dot{x},\ \Delta \theta,\ \Delta \dot{\theta}]^\top \).

    Ce que on observe:

    - L’erreur de position latérale \( \Delta x(t) \), initialement à 1 m, présente un léger dépassement (jusqu’à ~1.2 m à \( t = 1\,s \)), avant de converger vers zéro en environ **4 secondes**.
    - L’angle \( \Delta \theta(t) \) reste bien à l’intérieur de la limite \( \pm \pi/2 \).
    - La commande \( \Delta \phi(t) \) respecte également la contrainte physique \( |\Delta \phi(t)| < \pi/2 \).

    Cette approche par placement de pôles permet une stabilisation rapide et efficace du système, tout en respectant les contraintes dynamiques et physiques.

    """
    )
    return


@app.cell
def _(A_lat, B_lat):
    def _():
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.signal import place_poles, StateSpace, lsim


        desired_poles = [-2, -2.8, -3.1, -3.7]
        K_pp = place_poles(A_lat, B_lat, desired_poles).gain_matrix
        print("K_pp =", K_pp)

        A_cl = A_lat - B_lat @ K_pp


        t = np.linspace(0, 20, 1000)
        x0 = np.array([1.0, 0.0, 0.1, 0.0])
        sys_cl = StateSpace(A_cl, np.zeros((4, 1)), np.eye(4), np.zeros((4, 1)))
        _, y, _ = lsim(sys_cl, U=np.zeros_like(t), T=t, X0=x0)


        delta_x = y[:, 0]
        delta_theta = y[:, 2]
        delta_phi = - (y @ K_pp.T).flatten()

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(t, delta_x, label=r"$\Delta x(t)$")
        plt.axhline(0.1, color='gray', linestyle='--')
        plt.axhline(-0.1, color='gray', linestyle='--')
        plt.title("Erreur de position latérale")
        plt.xlabel("Temps [s]")
        plt.ylabel("Δx [m]")
        plt.grid()
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(t, delta_phi, label=r"$\Delta \phi(t)$")
        plt.axhline(np.pi/2, color='r', linestyle='--', label='±π/2')
        plt.axhline(-np.pi/2, color='r', linestyle='--')
        plt.title("Commande d'inclinaison")
        plt.xlabel("Temps [s]")
        plt.ylabel("ϕ [rad]")
        plt.grid()
        plt.legend()

        plt.tight_layout()
        return plt.show()


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controller Tuned with Optimal Control

    Using optimal, find a gain matrix $K_{oc}$ that satisfies the same set of requirements that the one defined using pole placement.

    Explain how you find the proper design parameters!
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Validation

    Test the two control strategies (pole placement and optimal control) on the "true" (nonlinear) model and check that they achieve their goal. Otherwise, go back to the drawing board and tweak the design parameters until they do!
    """
    )
    return


if __name__ == "__main__":
    app.run()
