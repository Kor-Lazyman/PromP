from ProMP_inner import *
from ProMP_optimizer import OuterLoopOptimizer
import config_folders
import config_RL
import config_SimPy
from GymWrapper import *
from Def_Scenarios import *
from model import *

if __name__ == '__main__':
    env = GymInterface()
    beta = BETA
    action_dim = [len(ACTION_SPACE) for _ in range(env.mat_count)]
    state_dim = len(env.reset())

    # Meta-parameters (already set up as mentioned)
    num_iterations = 1000  # Number of outer loop iterations
    batch_size = 16        # Number of tasks per batch
    convergence_threshold = 1e-4
    
    # Initialize meta-policy parameters θ
    theta = ActorCritic(state_dim, action_dim)
    
    # Outer loop - ProMP algorithm
    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}/{num_iterations}")
        
        # Step 3: Sample batch of tasks Ti ~ ρ(T)
        scenarios = create_scenarios()
        task_batch, test_batch = split_scenarios(scenarios)
        
        # Initialize gradient accumulator for meta-update
        
        inner_optim = []
        post_updated_trajectories = []
        # Step 7: For all Ti ~ ρ(T)
        for n in range(META_PARAMETER_UPDATE):
            meta_gradients = []
            for task_idx in range(1):
                env.reset()
                env.scenario = task_batch[task_idx]

                if n == 0:
                    # Step 8: Sample pre-update trajectories using current policy πθ            
                    # Step 9: Compute adapted parameters θ'0,i using inner loop
                    # (inner loop is already completed as mentioned)
                    inner = inner_loop(theta, env, 1, ALPHA)
                    
                    # Step 10: Sample post-update trajectories using adapted policy πθ'
                    theta_old, post_updated_trajectorie = inner._first_optimization()

                    inner_optim.append(OuterLoopOptimizer(ALPHA, theta_old))
                    post_updated_trajectories.append(post_updated_trajectorie)
                
                meta_gradients.append(inner_optim[task_idx].optimizer(post_updated_trajectories[task_idx], theta, state_dim, action_dim))
            
            # Step 11: Meta-update θ ← θ + β∑Ti ∇θJ^ProMP_Ti(θ)
            # Average gradients across all tasks in the batch

            total_gradients = [
                                beta*sum(layer_grads)
                                for layer_grads in zip(*meta_gradients)
                                ]
            with torch.no_grad():
                for param, grad in zip(theta.parameters(), total_gradients):
                    param.data.add_(grad)  # θ ← θ + α∇θ
        
    
    print("Training completed!")
    
    # Save final meta-policy
    torch.save(theta, 'meta_policy_promp.pth')