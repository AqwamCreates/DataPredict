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
| MonteCarloControl                                                                  | 1                         | Both                              | On-Policy              | Yes          | No           | No              | Yes                   | Yes                     |
| OffPolicyMonteCarloControl                                                         | 1                         | Both                              | Off-Policy             | Yes          | No           | No              | Yes                   | Yes                     |
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
   - **Characteristics**: Similar to the Double Deep Q-Learning method but applied to the State-Action-Reward-State-Action framework.
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

10. **REINFORCE**:
    - **Characteristics**: A Monte Carlo method optimizing the policy based on complete returns.
    - **Advantages**: Straightforward implementation for policy optimization.
    - **Disadvantages**: High variance in updates can lead to slow convergence.

11. **Monte Carlo Control**
   - **Characteristics**: Uses sample returns to estimate the optimal policy and action-value function without requiring knowledge of the environment dynamics.
   - **Advantages**:
     - Effective for episodic tasks where the episode's return can be fully computed.
     - Can directly approximate the optimal policy through iterative updates.
   - **Disadvantages**:
     - Requires complete episodes for updates, limiting its application to non-episodic tasks.
     - High variance in returns can slow convergence.

12. **Off-Policy Monte Carlo Control**
   - **Characteristics**: Learns an optimal policy (target policy) using data generated by a different policy (behavior policy).
   - **Advantages**:
     - Allows for more diverse exploration since the behavior policy can differ from the target policy.
     - More sample efficient compared to on-policy Monte Carlo control when combined with importance sampling techniques.
   - **Disadvantages**:
     - Importance sampling weights can lead to high variance in updates.
     - Requires careful balancing of exploration in the behavior policy to ensure sufficient coverage of the state-action space.

13. **Vanilla Policy Gradient**:
    - **Characteristics**: Estimates policy gradients using both actor and critic.
    - **Advantages**: More stable than REINFORCE due to variance reduction from the critic.
    - **Disadvantages**: Still suffers from high variance in gradient estimates.

14. **Actor-Critic**:
    - **Characteristics**: Combines value function approximation with policy optimization.
    - **Advantages**: Provides more stable learning through actor and critic interaction.
    - **Disadvantages**: Requires careful tuning to balance actor and critic updates.

15. **Advantage Actor-Critic**:
    - **Characteristics**: Uses advantages to improve the learning signal for the actor.
    - **Advantages**: Reduces variance in the policy gradient estimates.
    - **Disadvantages**: Accurate advantage estimation can be challenging.

16. **Asynchronous Advantage Actor-Critic**:
    - **Characteristics**: Employs multiple agents in parallel to speed up training.
    - **Advantages**: Faster convergence due to diverse exploration.
    - **Disadvantages**: Increased implementation complexity.

17. **Proximal Policy Optimization**:
    - **Characteristics**: Clipped objective function to ensure stable policy updates.
    - **Advantages**: Balances exploration and exploitation effectively.
    - **Disadvantages**: Sensitive to the clipping range and other hyperparameters.

18. **Proximal Policy Optimization with Clipped Objective**:
    - **Characteristics**: An extension of PPO focused on stability.
    - **Advantages**: Helps prevent large policy updates that can destabilize learning.
    - **Disadvantages**: Requires careful parameter tuning for optimal performance.
