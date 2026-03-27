from ursina import Ursina, Entity, EditorCamera, DirectionalLight, Sky, color, Vec3
from ursina.shaders import lit_with_shadows_shader
import random
import math

try:
    from LAB1.model.Cube import Cube
    from LAB1.model.Particles import Particles
except ModuleNotFoundError:
    from model.Cube import Cube
    from model.Particles import Particles


ENTROPY_MODE = 'both'  # 'boltzmann', 'shannon', or 'both'


def _lerp(a, b, t):
    return a + (b - a) * t


def entropy_to_color(value):
    value = max(0.0, min(1.0, value))
    value = value ** 0.6
    stops = [
        (0.0, (32, 48, 96)),    # deep blue
        (0.25, (20, 164, 196)), # cyan
        (0.5, (96, 196, 72)),   # green
        (0.75, (240, 190, 40)), # amber
        (1.0, (220, 56, 32)),   # red
    ]

    for idx in range(len(stops) - 1):
        left_t, left_rgb = stops[idx]
        right_t, right_rgb = stops[idx + 1]
        if left_t <= value <= right_t:
            local_t = (value - left_t) / (right_t - left_t)
            r = int(_lerp(left_rgb[0], right_rgb[0], local_t))
            g = int(_lerp(left_rgb[1], right_rgb[1], local_t))
            b = int(_lerp(left_rgb[2], right_rgb[2], local_t))
            alpha = _lerp(0.25, 0.65, value)
            return color.rgba32(r, g, b, int(alpha * 255))

    r, g, b = stops[-1][1]
    return color.rgba32(r, g, b, int(0.4 * 255))


def render_cube(cube_obj: Cube):
    n = cube_obj.n
    s = cube_obj.side_length / n
    subcube_entities = []
    subcube_outline_entities = []

    Entity(
        model='wireframe_cube',
        color=color.black,
        position=(0, 0, 0),
        scale=(cube_obj.side_length, cube_obj.side_length, cube_obj.side_length)
    )

    for i, _sub in enumerate(cube_obj.sub_cubes):
        x = i % n
        y = (i // n) % n
        z = i // (n * n)

        x_pos = (x - (n - 1) / 2) * s
        y_pos = (y - (n - 1) / 2) * s
        z_pos = (z - (n - 1) / 2) * s

        subcube_entity = Entity(
            model='cube',
            color=entropy_to_color(0.0),
            position=(x_pos, y_pos, z_pos),
            scale=(s * 0.95, s * 0.95, s * 0.95),
            double_sided=True,
            collider='box'
        )
        subcube_entity.set_transparency(True)
        subcube_entity.depth_write = False
        subcube_entities.append(subcube_entity)

        outline_entity = Entity(
            model='wireframe_cube',
            color=entropy_to_color(0.0),
            position=(x_pos, y_pos, z_pos),
            scale=(s * 0.95, s * 0.95, s * 0.95)
        )
        subcube_outline_entities.append(outline_entity)

    return subcube_entities, subcube_outline_entities


def update():
    c.move_particles()
    particles_list = c.particles
    collided_particles = set()

    for i in range(len(particles_list)):
        for j in range(i + 1, len(particles_list)):
            p1, p2 = particles_list[i], particles_list[j]
            if p1.collides_with(p2):
                p1.resolve_collision(p2)
                collided_particles.add(p1)
                collided_particles.add(p2)

    entropy_values = c.compute_subcube_entropies(method=ENTROPY_MODE)
    for idx, subcube_entity in enumerate(subcube_entities):
        if idx < len(entropy_values):
            entropy_color = entropy_to_color(entropy_values[idx])
            subcube_entity.color = entropy_color
            subcube_outline_entities[idx].color = entropy_color

    half_size = c.side_length / 2
    for p, ent in particle_entities:
        # Bounce while accounting for particle radius.
        if abs(p.position.x) + p.radius > half_size:
            p.velocity.x *= -1
            p.position.x = math.copysign(half_size - p.radius, p.position.x)
        if abs(p.position.y) + p.radius > half_size:
            p.velocity.y *= -1
            p.position.y = math.copysign(half_size - p.radius, p.position.y)
        if abs(p.position.z) + p.radius > half_size:
            p.velocity.z *= -1
            p.position.z = math.copysign(half_size - p.radius, p.position.z)
            
        ent.position = p.position
        ent.color = color.yellow if p in collided_particles else color.red


if __name__ == "__main__":
    app = Ursina()

    c = Cube(side_length=4, outline_color='orange', n=3, particule_number=0)
    subcube_entities, subcube_outline_entities = render_cube(c)

    particle_entities = []
    particle_count = 80
    particle_radius = 0.09

    # Spawn particles with enough density/velocity to make collisions visible.
    for _ in range(particle_count):
        pos = Vec3(
            random.uniform(-c.side_length/2, c.side_length/2),
            random.uniform(-c.side_length/2, c.side_length/2),
            random.uniform(-c.side_length/2, c.side_length/2)
        )
        vel = Vec3(
            random.uniform(-0.03, 0.03),
            random.uniform(-0.03, 0.03),
            random.uniform(-0.03, 0.03)
        )
        p = Particles(position=pos, velocity=vel/10, radius=particle_radius)
        c.add_particle(p)
        
        ent = Entity(model='sphere', color=color.red, scale=particle_radius * 2, position=pos)
        particle_entities.append((p, ent))

    DirectionalLight(y=2, z=3)
    Sky()
    EditorCamera()  # mouse orbit / zoom

    app.run()
