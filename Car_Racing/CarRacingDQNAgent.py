import random
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam

class CarRacingDQNAgent:
    def __init__(self):
        """Initialize the DQN Agent for CarRacing-v2."""
        self.action_space = [0, 1, 2, 3, 4]  # ✅ Discrete actions for CarRacing-v2

        self.frame_stack_num = 3
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95  
        self.epsilon = 1.0  
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9999
        self.learning_rate = 0.001
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        """Build a CNN-based DQN model."""
        model = Sequential()
        model.add(Input(shape=(96, 96, 3)))  
        model.add(Conv2D(32, (5, 5), strides=2, activation='relu', padding="same"))  
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), strides=2, activation='relu', padding="same"))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))  
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(len(self.action_space), activation="softmax"))  
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=self.learning_rate, clipnorm=1.0))
        return model

    def update_target_model(self):
        """Update target model weights."""
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        """Store experiences in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Select an action using an ε-greedy policy."""
        state = np.expand_dims(state, axis=0)  
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_space)  
        else:
            q_values = self.model.predict(state, verbose=0)
            return np.argmax(q_values[0])  

    def replay(self, batch_size):
        """Train the model using stored experiences."""
        if len(self.memory) < batch_size:
            return  
        
        minibatch = random.sample(self.memory, batch_size)
        train_state, train_target = [], []
        
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0]
            if done:
                target[action] = reward
            else:
                future_q = self.target_model.predict(np.expand_dims(next_state, axis=0), verbose=0)[0]
                target[action] = reward + self.gamma * np.amax(future_q)

            train_state.append(state)
            train_target.append(target)

        self.model.fit(np.array(train_state), np.array(train_target), epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """Load model weights."""
        self.model.load_weights(name)
        self.update_target_model()

    def save(self, name):
        """Save the model."""
        filename = name if name.endswith(".h5") else name + ".h5"
        self.target_model.save(filename)
        print(f"Model saved to {filename}")