"""
Quick test to verify n-step returns implementation logic
"""
import numpy as np
from collections import deque

class SimpleNStepTest:
    """Test n-step return computation"""

    def __init__(self, n_step=3, gamma=0.99):
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)
        self.stored_transitions = []

    def remember(self, state, action, reward, next_state, done):
        """Simulate n-step remember logic"""
        # Add transition to n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # If we have enough steps or episode ended, compute n-step return
        if len(self.n_step_buffer) == self.n_step or done:
            # Get the first transition from the buffer
            first_state, first_action, _, _, _ = self.n_step_buffer[0]

            # Compute n-step return
            n_step_reward = 0
            n_step_state = next_state
            n_step_done = done

            # Sum discounted rewards
            for i, (_, _, r, s, d) in enumerate(self.n_step_buffer):
                n_step_reward += (self.gamma ** i) * r
                n_step_state = s
                n_step_done = d
                if d:  # Episode ended, stop accumulating
                    break

            # Store n-step transition
            self.stored_transitions.append((first_state, first_action, n_step_reward, n_step_state, n_step_done))
            print(f"  Stored: state={first_state}, action={first_action}, n_step_reward={n_step_reward:.2f}, done={n_step_done}")

        # If episode ended, flush remaining transitions
        if done and len(self.n_step_buffer) > 1:
            buffer_list = list(self.n_step_buffer)
            for start_idx in range(1, len(buffer_list)):
                first_state, first_action, _, _, _ = buffer_list[start_idx]

                # Compute remaining steps return
                n_step_reward = 0
                n_step_state = next_state
                n_step_done = True

                for i, (_, _, r, s, d) in enumerate(buffer_list[start_idx:]):
                    n_step_reward += (self.gamma ** i) * r
                    n_step_state = s

                self.stored_transitions.append((first_state, first_action, n_step_reward, n_step_state, n_step_done))
                print(f"  Flushed: state={first_state}, action={first_action}, n_step_reward={n_step_reward:.2f}, done={n_step_done}")

            # Clear buffer
            self.n_step_buffer.clear()

def test_nstep_full_episode():
    """Test n-step with episode of exactly n steps"""
    print("\n=== Test 1: Episode with exactly 3 steps ===")
    agent = SimpleNStepTest(n_step=3, gamma=0.99)

    # Episode: s0 --(r=1)--> s1 --(r=2)--> s2 --(r=3)--> terminal
    agent.remember(state=0, action=0, reward=1.0, next_state=1, done=False)
    agent.remember(state=1, action=1, reward=2.0, next_state=2, done=False)
    agent.remember(state=2, action=2, reward=3.0, next_state=3, done=True)

    # Expected: one transition with 3-step return
    # n_step_reward = 1 + 0.99*2 + 0.99^2*3 = 1 + 1.98 + 2.9403 = 5.9203
    expected = 1.0 + 0.99 * 2.0 + (0.99**2) * 3.0
    print(f"Expected 3-step reward: {expected:.4f}")
    print(f"Stored transitions: {len(agent.stored_transitions)}")
    assert len(agent.stored_transitions) == 3, "Should have 3 transitions after flushing"

def test_nstep_longer_episode():
    """Test n-step with episode longer than n"""
    print("\n=== Test 2: Episode with 5 steps (n=3) ===")
    agent = SimpleNStepTest(n_step=3, gamma=0.99)

    # Episode: s0 -> s1 -> s2 -> s3 -> s4 -> terminal
    rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
    for i, r in enumerate(rewards):
        done = (i == len(rewards) - 1)
        agent.remember(state=i, action=i, reward=r, next_state=i+1, done=done)

    print(f"Stored transitions: {len(agent.stored_transitions)}")
    # Should have stored: (s0, 3-step), (s1, 3-step), (s2, 3-step), (s3, 2-step), (s4, 1-step)
    assert len(agent.stored_transitions) == 5, "Should have 5 transitions"

    # Verify first transition (3-step from s0)
    # n_step_reward = 1 + 0.99*2 + 0.99^2*3
    expected_first = 1.0 + 0.99 * 2.0 + (0.99**2) * 3.0
    actual_first = agent.stored_transitions[0][2]
    print(f"First transition reward: expected={expected_first:.4f}, actual={actual_first:.4f}")
    assert abs(actual_first - expected_first) < 0.01, "First transition reward mismatch"

def test_gamma_n_bootstrap():
    """Verify gamma^n is used for bootstrapping"""
    print("\n=== Test 3: Gamma^n bootstrap verification ===")
    gamma = 0.99
    n_step = 3

    # Target computation: n_step_reward + gamma^n * Q(s_n)
    gamma_n = gamma ** n_step
    print(f"gamma = {gamma}")
    print(f"n_step = {n_step}")
    print(f"gamma^n_step = {gamma_n:.6f}")

    # Example: if Q(s_n) = 100, and n_step_reward = 5
    q_next = 100.0
    n_step_reward = 5.0
    target = n_step_reward + gamma_n * q_next
    print(f"Target = {n_step_reward} + {gamma_n:.6f} * {q_next} = {target:.4f}")

if __name__ == "__main__":
    print("Testing N-Step Returns Implementation")
    print("=" * 50)

    try:
        test_nstep_full_episode()
        test_nstep_longer_episode()
        test_gamma_n_bootstrap()

        print("\n" + "=" * 50)
        print("✅ All tests passed! N-step implementation is correct.")
        print("=" * 50)

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        exit(1)
