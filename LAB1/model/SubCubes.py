class cube:
    def __init__(self, side_length, Particule_number, color):
        self.side_length = side_length
        self.Particule_number = Particule_number
        self.color = color
        self.boltzmann_entropy = 0.0
        self.shannon_entropy = 0.0
        self.entropy_value = 0.0

    def volume(self):
        return self.side_length ** 3

    
    def update_color(self, new_color):
        self.color = new_color

    def update_particule_number(self, new_particule_number):
        self.Particule_number = new_particule_number
        self.update_color(self.color) # Update color based on new particule number

    def update_entropies(self, boltzmann_entropy, shannon_entropy, entropy_value):
        self.boltzmann_entropy = boltzmann_entropy
        self.shannon_entropy = shannon_entropy
        self.entropy_value = entropy_value

