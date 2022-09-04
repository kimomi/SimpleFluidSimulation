# ref https://lucasschuermann.com/writing/implementing-sph-in-2d
import taichi as ti

ti.init(arch=ti.gpu)

NUM_PARTICLES = 500
PARTICLES_X_SIZE = 25

H = 16.0 # kernel radius
EPS = H # boundary epsilon
HSQ = H * H
WIDTH = 900
HEIGHT = 900
MASS = 1.2
GAS_CONST = 2000.0 # const for equation of state
REST_DENS = 300.0 # rest density
VISC = 50.0 # larger -> viscous
G = ti.Vector.field(2, shape=(), dtype=ti.f32)
G[None] = ti.Vector([0.0, -9.8])
DT = 0.00045
BOUND_DAMPING = -0.5

POLY6 = 4.0 / (ti.math.pi * ti.math.pow(H, 8))
SPIKY_GRAD = -10.0 / (ti.math.pi * ti.math.pow(H, 5))
VISC_LAP =  40.0 / (ti.math.pi * ti.math.pow(H, 5))

position = ti.Vector.field(2, shape=(NUM_PARTICLES), dtype=ti.f32)
velocity = ti.Vector.field(2, shape=(NUM_PARTICLES), dtype=ti.f32)
force = ti.Vector.field(2, shape=(NUM_PARTICLES), dtype=ti.f32)
density = ti.field(shape=(NUM_PARTICLES), dtype=ti.f32)
pressure = ti.field(shape=(NUM_PARTICLES), dtype=ti.f32)

# set water initialize shape
@ti.kernel
def init_SPH():
    for i in position:
        x = WIDTH / 2 + (i % PARTICLES_X_SIZE - PARTICLES_X_SIZE / 2) * H * 0.99
        y = EPS + H * 0.99 * ti.math.floor(i / PARTICLES_X_SIZE) + 100
        position[i] = ti.Vector([x, y])
        velocity[i] = ti.Vector([0.0, 0.0])
        force[i] = ti.Vector([0.0, 0.0])
        density[i] = 0
        pressure[i] = 0

@ti.kernel
def update():
    # compute density pressure
    for i in position:
        density[i] = 0
        for j in range(NUM_PARTICLES):
            sqr = (position[j] - position[i]).norm_sqr()

            if sqr < HSQ:
                density[i] += MASS * POLY6 * ti.math.pow(HSQ - sqr, 3)
        pressure[i] = GAS_CONST * (density[i] - REST_DENS)

    # compute forces
    for i in position:
        fpress = ti.Vector([0.0, 0.0])
        fvisc = ti.Vector([0.0, 0.0])
        for j in range(NUM_PARTICLES):
            if i == j:
                continue

            rij = position[j] - position[i]
            r = rij.norm()

            if r < 1e-10: # aviod same pos
                rij = ti.Vector([ti.random() + 0.001, ti.random() + 0.001])
                r = 1e-10

            if r < H:
                fpress += -rij.normalized() * MASS * (pressure[i] + pressure[j]) / (2.0 * density[j]) * SPIKY_GRAD * ti.math.pow(H - r, 3)
                fvisc += VISC * MASS * (velocity[j] - velocity[i]) / density[j] * VISC_LAP * (H - r)
        fgrav = G[None] * MASS / density[i]
        force[i] = fpress + fvisc + fgrav
    
    # move by force
    for i in position:
        velocity[i] += DT * force[i] / density[i]
        position[i] += DT * velocity[i]
        # print(position[i])

        if position[i][0] < EPS:
            velocity[i][0] *= BOUND_DAMPING
            position[i][0] = EPS
        
        if position[i][0] + EPS >= WIDTH:
            velocity[i][0] *= BOUND_DAMPING
            position[i][0] = WIDTH - EPS

        if position[i][1] <= EPS:
            velocity[i][1] *= BOUND_DAMPING
            position[i][1] = EPS
        
        if position[i][1] + EPS >= HEIGHT:
            velocity[i][1] *= BOUND_DAMPING
            position[i][1] = HEIGHT - EPS

def render(g : ti.GUI):
    g.circles(position.to_numpy() / WIDTH, color=0xFFFFFF, radius=4)

gui = ti.GUI("sph", res=(WIDTH, HEIGHT))

init_SPH()

while gui.running:
    gui.get_event()
    if gui.is_pressed(ti.GUI.SPACE):
        init_SPH()
    elif gui.is_pressed(ti.GUI.LMB):
        x, y = gui.get_cursor_pos()
        G[None] = ti.Vector([x * 2 - 1, y * 2 - 1]).normalized() * 9.8
    update()
    render(gui)
    gui.show()