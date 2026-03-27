class Particles:
    def __init__(self, position, velocity, radius=0.075):
        self.position = position
        self.velocity = velocity
        self.radius=radius
    
    def move(self):
        self.position += self.velocity

    def collides_with(self, other):
        distance = (self.position - other.position)
        dist_sq=distance.x**2 + distance.y**2 + distance.z**2
        radius_sum = self.radius + other.radius
        return dist_sq <= radius_sum ** 2
    
    def resolve_collision(self, other):
        diff = self.position - other.position
        dist = (diff.x**2 + diff.y**2 + diff.z**2) ** 0.5
        if dist == 0:
            return  # avoid division by zero

        # Collision normal (unit vector)
        nx, ny, nz = diff.x / dist, diff.y / dist, diff.z / dist

        # Relative velocity along the normal
        dvx = self.velocity.x - other.velocity.x
        dvy = self.velocity.y - other.velocity.y
        dvz = self.velocity.z - other.velocity.z
        dot = dvx * nx + dvy * ny + dvz * nz

        if dot >= 0:
            return  # already separating

        # Exchange impulse (equal mass elastic collision)
        self.velocity.x -= dot * nx
        self.velocity.y -= dot * ny
        self.velocity.z -= dot * nz
        other.velocity.x += dot * nx
        other.velocity.y += dot * ny
        other.velocity.z += dot * nz

        # Positional correction (prevent overlap)
        overlap = (self.radius + other.radius - dist) / 2
        self.position.x += nx * overlap
        self.position.y += ny * overlap
        self.position.z += nz * overlap
        other.position.x -= nx * overlap
        other.position.y -= ny * overlap
        other.position.z -= nz * overlap