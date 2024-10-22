# Deep Reinforcement Learning Algorithms Properties

| Algorithm                                                                          | Number Of Neural Networks | Temporal Difference / Monte-Carlo | On-Policy / Off-Policy | Q-Values     | V-Values     | Policy-Gradient | Discrete Action Space | Continuous Action Space |
|------------------------------------------------------------------------------------|---------------------------|-----------------------------------|------------------------|--------------|--------------|-----------------|-----------------------|-------------------------|
| Deep Q Learning                                                                    | 1                         | Temporal Difference               | Off-Policy             | Yes          | No           | No              | Yes                   | No                      |
| Double Deep Q Learning V1 (Randomly Chosen Network)                                | 1 (2 Model Parameters)    | Temporal Difference               | Off-Policy             | Yes          | No           | No              | Yes                   | No                      |
| Double Deep Q Learning V2 (Target Network)                                         | 1 (2 Model Parameters)    | Temporal Difference               | Off-Policy             | Yes          | No           | No              | Yes                   | No                      |
| Deep State-Action-Reward-State-Action                                              | 1                         | Temporal Difference               | On-Policy              | Yes          | No           | No              | Yes                   | No                      |
| Double Deep State-Action-Reward-State-Action V1 (Randomly Chosen Network)          | 1 (2 Model Parameters)    | Temporal Difference               | On-Policy              | Yes          | No           | No              | Yes                   | No                      |
| Double Deep State-Action-Reward-State-Action V2 (Target Network)                   | 1 (2 Model Parameters)    | Temporal Difference               | On-Policy              | Yes          | No           | No              | Yes                   | No                      |
| Deep Expected State-Action-Reward-State-Action                                     | 1                         | Temporal Difference               | On-Policy              | Yes          | No           | No              | Yes                   | No                      |
| Double Deep Expected State-Action-Reward-State-Action V1 (Randomly Chosen Network) | 1 (2 Model Parameters)    | Temporal Difference               | On-Policy              | Yes          | No           | No              | Yes                   | No                      |
| Double Deep Expected State-Action-Reward-State-Action V2 (Target Network)          | 1 (2 Model Parameters)    | Temporal Difference               | On-Policy              | Yes          | No           | No              | Yes                   | No                      |
| REINFORCE                                                                          | 1                         | Both                              | On-Policy              | No           | Yes          | Yes             | Yes                   | Yes                     |
| Vanilla Policy Gradient                                                            | 2 (Actor + Critic)        | Both                              | On-Policy              | Yes (Actor)  | Yes (Critic) | Yes             | Yes                   | Yes                     |
| Actor-Critic                                                                       | 2 (Actor + Critic)        | Both                              | On-Policy              | Yes (Actor)  | Yes (Critic) | Yes             | Yes                   | Yes                     |
| Advantage Actor-Critic                                                             | 2 (Actor + Critic)        | Both                              | On-Policy              | Yes (Actor)  | Yes (Critic) | Yes             | Yes                   | Yes                     |
| Asynchronous Advantage Actor-Critic                                                | 2 (Actor + Critic)        | Both                              | On-Policy              | Yes (Actor)  | Yes (Critic) | Yes             | Yes                   | Yes                     |
| Proximal Policy Optimization                                                       | 2 (Actor + Critic)        | Both                              | On-Policy              | Yes (Actor)  | Yes (Critic) | Yes             | Yes                   | Yes                     |
| Proximal Policy Optimization with Clipped Objective                                | 2 (Actor + Critic)        | Both                              | On-Policy              | Yes (Actor)  | Yes (Critic) | Yes             | Yes                   | Yes                     |

## Additional Notes:
1. **Deep Q Learning**:
   - **Characteristics**: Uses a neural network to approximate Q-values.
   - **Advantages**: Simple to implement; effective for discrete action spaces.
   - **Disadvantages**: Can struggle with stability and may overestimate Q-values.

2. **Double Deep Q Learning V1 (Randomly Chosen Network)**:
   - **Characteristics**: Mitigates overestimation by randomly choosing one of two sets of model parameters for updates.
   - **Advantages**: Reduces bias from the greedy policy during action selection.
   - **Disadvantages**: Still sensitive to hyperparameter tuning.

3. **Double Deep Q Learning V2 (Target Network)**:
   - **Characteristics**: Uses a separate target network for stable Q-value updates.
   - **Advantages**: Further reduces overestimation bias and improves training stability.
   - **Disadvantages**: More complex due to the need for synchronization of networks.

4. **Deep State-Action-Reward-State-Action**:
   - **Characteristics**: An extension that uses a neural network to estimate action values based on the current policy.
   - **Advantages**: Suitable for environments with varying action rewards.
   - **Disadvantages**: Performance can degrade with insufficient exploration.

5. **Double Deep State-Action-Reward-State-Action V1 (Randomly Chosen Network)**:
   - **Characteristics**: Similar to the double DQ learning method but applied to the State-Action-Reward-State-Action framework.
   - **Advantages**: Helps address overestimation in policy evaluation.
   - **Disadvantages**: Increased complexity in choosing which parameters to update.

6. **Double Deep State-Action-Reward-State-Action V2 (Target Network)**:
   - **Characteristics**: Incorporates a target network to stabilize training.
   - **Advantages**: Offers improved performance by decoupling the Q-value updates.
   - **Disadvantages**: Requires additional resources to maintain the target network.

7. **Deep Expected State-Action-Reward-State-Action**:
   - **Characteristics**: Considers the expected value of future states for more robust action evaluation.
   - **Advantages**: More stable than traditional Q-learning methods.
   - **Disadvantages**: Sensitive to the choice of exploration strategies.

8. **Double Deep Expected State-Action-Reward-State-Action V1 (Randomly Chosen Network)**:
   - **Characteristics**: Enhances expected State-Action-Reward-State-Action by mitigating overestimation bias through random selection of parameters.
   - **Advantages**: Reduces variance in Q-value estimates.
   - **Disadvantages**: May not always provide optimal exploration.

9. **Double Deep Expected State-Action-Reward-State-Action V2 (Target Network)**:
   - **Characteristics**: Utilizes a target network to improve stability and performance.
   - **Advantages**: Significantly enhances the stability of Q-value updates.
   - **Disadvantages**: Increases computational complexity.

10. **Vanilla Policy Gradient**:
    - **Characteristics**: Estimates policy gradients using both actor and critic.
    - **Advantages**: More stable than REINFORCE due to variance reduction from the critic.
    - **Disadvantages**: Still suffers from high variance in gradient estimates.

11. **REINFORCE**:
    - **Characteristics**: A Monte Carlo method optimizing the policy based on complete returns.
    - **Advantages**: Straightforward implementation for policy optimization.
    - **Disadvantages**: High variance in updates can lead to slow convergence.

12. **Actor-Critic**:
    - **Characteristics**: Combines value function approximation with policy optimization.
    - **Advantages**: Provides more stable learning through actor and critic interaction.
    - **Disadvantages**: Requires careful tuning to balance actor and critic updates.

13. **Advantage Actor-Critic**:
    - **Characteristics**: Uses advantages to improve the learning signal for the actor.
    - **Advantages**: Reduces variance in the policy gradient estimates.
    - **Disadvantages**: Accurate advantage estimation can be challenging.

14. **Asynchronous Advantage Actor-Critic**:
    - **Characteristics**: Employs multiple agents in parallel to speed up training.
    - **Advantages**: Faster convergence due to diverse exploration.
    - **Disadvantages**: Increased implementation complexity.

15. **Proximal Policy Optimization**:
    - **Characteristics**: Clipped objective function to ensure stable policy updates.
    - **Advantages**: Balances exploration and exploitation effectively.
    - **Disadvantages**: Sensitive to the clipping range and other hyperparameters.

16. **Proximal Policy Optimization with Clipped Objective**:
    - **Characteristics**: An extension of PPO focused on stability.
    - **Advantages**: Helps prevent large policy updates that can destabilize learning.
    - **Disadvantages**: Requires careful parameter tuning for optimal performance.
