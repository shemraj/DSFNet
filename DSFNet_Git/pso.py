class PSO:
    def __init__(self, model, learning_rate, weight_decay, beta1, beta2, epsilon, swarm_size):
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.swarm_size = swarm_size

        # Initialize particle positions and velocities
        self.positions = []
        self.velocities = []

        for param in self.model.parameters():
            position = torch.randn_like(param.data)
            velocity = torch.zeros_like(param.data)
            self.positions.append(position)
            self.velocities.append(velocity)

        # Initialize Adam optimizer statistics
        self.m_t = [torch.zeros_like(param.data) for param in self.model.parameters()]
        self.v_t = [torch.zeros_like(param.data) for param in self.model.parameters()]

    def update(self):
        for i, param in enumerate(self.model.parameters()):
            # Compute the gradient
            gradient = param.grad.data

            # Particle swarm update
            velocity = self.velocities[i] + self.positions[i] * gradient
            position = param.data - self.learning_rate * velocity

            # Update position and velocity
            self.positions[i] = position
            self.velocities[i] = velocity

            # Apply weight decay
            if self.weight_decay > 0:
                position.mul_(1 - self.learning_rate * self.weight_decay)

            # Adam optimizer update
            self.m_t[i] = self.beta1 * self.m_t[i] + (1 - self.beta1) * gradient
            self.v_t[i] = self.beta2 * self.v_t[i] + (1 - self.beta2) * gradient * gradient

            m_hat = self.m_t[i] / (1 - self.beta1)
            v_hat = self.v_t[i] / (1 - self.beta2)

            param.data.copy_(position - self.learning_rate * m_hat / (torch.sqrt(v_hat) + self.epsilon))
