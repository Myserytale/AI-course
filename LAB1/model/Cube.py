from .SubCubes import cube as SubCube
from .Particles import Particles
import math

class Cube:
    def __init__(self, side_length, outline_color, n, particule_number=0):
        if n <= 0:
            raise ValueError("n must be greater than 0")

        self.side_length = side_length
        self.color = outline_color
        self.n = n
        self.sub_cubes = []
        self.particles = []
        self.sub_cube_side_length = self.side_length / self.n
        self._build_sub_cubes(particule_number)

    def _build_sub_cubes(self, particule_number):
        total_sub_cubes = self.n * self.n * self.n
        for _ in range(total_sub_cubes):
            self.sub_cubes.append(SubCube(self.sub_cube_side_length, particule_number, self.color))
    
    def add_sub_cube(self, sub_cube):
        if isinstance(sub_cube, SubCube):
            self.sub_cubes.append(sub_cube)
    
    def add_particle(self, particle):
        if isinstance(particle, Particles):
            self.particles.append(particle)
    
    def total_volume(self):
        return sum(sub_cube.volume() for sub_cube in self.sub_cubes)
    
    def move_particles(self):
        for particle in self.particles:
            particle.move()

    def _clamp_index(self, value):
        return max(0, min(self.n - 1, value))

    def _particle_subcube_index(self, particle):
        half_side = self.side_length / 2
        s = self.sub_cube_side_length

        x_idx = self._clamp_index(int((particle.position.x + half_side) / s))
        y_idx = self._clamp_index(int((particle.position.y + half_side) / s))
        z_idx = self._clamp_index(int((particle.position.z + half_side) / s))

        return z_idx * self.n * self.n + y_idx * self.n + x_idx

    def _subcube_counts(self):
        counts = [0] * len(self.sub_cubes)
        for particle in self.particles:
            idx = self._particle_subcube_index(particle)
            counts[idx] += 1

        for sub_cube, count in zip(self.sub_cubes, counts):
            sub_cube.update_particule_number(count)

        return counts

    def compute_subcube_entropies(self, method='both'):
        method = method.lower()
        if method not in {'boltzmann', 'shannon', 'both'}:
            raise ValueError("method must be 'boltzmann', 'shannon', or 'both'")

        counts = self._subcube_counts()
        if not counts:
            return []

        total_particles = len(self.particles)
        max_boltzmann = math.log(total_particles + 1) if total_particles > 0 else 1.0

        shannon_terms = []
        for count in counts:
            if total_particles == 0 or count == 0:
                shannon_terms.append(0.0)
            else:
                p_i = count / total_particles
                shannon_terms.append(-p_i * math.log(p_i))

        # Theoretical maximum of -p ln(p) occurs at p = 1/e.
        max_shannon_term = 1.0 / math.e

        entropy_values = []
        for i, (sub_cube, count) in enumerate(zip(self.sub_cubes, counts)):
            boltzmann_entropy = math.log(count + 1)
            shannon_entropy = shannon_terms[i]

            boltzmann_norm = boltzmann_entropy / max_boltzmann if max_boltzmann else 0.0
            shannon_norm = shannon_entropy / max_shannon_term if max_shannon_term else 0.0

            boltzmann_norm = max(0.0, min(1.0, boltzmann_norm))
            shannon_norm = max(0.0, min(1.0, shannon_norm))

            if method == 'boltzmann':
                entropy_value = boltzmann_norm
            elif method == 'shannon':
                entropy_value = shannon_norm
            else:
                entropy_value = 0.5 * (boltzmann_norm + shannon_norm)

            sub_cube.update_entropies(boltzmann_entropy, shannon_entropy, entropy_value)
            entropy_values.append(entropy_value)

        return entropy_values
    
