
def plot_training_curves(iterations, policy_losses, critic_losses, returns, entropy_losses, plots_dir, save=False):
    """
    Plot and optionally save training curves.
    
    Args:
        iterations: List of iteration numbers
        policy_losses: List of policy loss values
        critic_losses: List of critic loss values
        returns: List of average returns
        entropy_losses: List of entropy loss values
        plots_dir: Directory to save plots
        save: Whether to save plots to disk
    """
    plt.figure(figsize=(12, 10))
    
    # Plot policy losses
    plt.subplot(2, 2, 1)
    plt.plot(iterations, policy_losses, 'b-')
    plt.title('Policy Losses')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot critic losses
    plt.subplot(2, 2, 2)
    plt.plot(iterations, critic_losses, 'r-')
    plt.title('Critic Losses')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot returns
    plt.subplot(2, 2, 3)
    plt.plot(iterations, returns, 'g-')
    plt.title('Average Returns')
    plt.xlabel('Iteration')
    plt.ylabel('Return')
    plt.grid(True)
    
    # Plot entropy losses
    plt.subplot(2, 2, 4)
    plt.plot(iterations, entropy_losses, 'm-')
    plt.title('Entropy Losses')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.tight_layout()
    
    if save:
        # Save individual plots
        save_individual_plots(iterations, policy_losses, critic_losses, returns, entropy_losses, plots_dir)
        
        # Save combined plot
        plt.savefig(f"{plots_dir}/combined_training_curves.png", dpi=300)
        
    plt.close()

def save_individual_plots(iterations, policy_losses, critic_losses, returns, entropy_losses, plots_dir):
    """Save individual plots for each metric."""
    
    # Policy loss plot
    plt.figure(figsize=(8, 6))
    plt.plot(iterations, policy_losses, 'b-')
    plt.title('Policy Losses')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(f"{plots_dir}/policy_losses.png", dpi=300)
    plt.close()
    
    # Critic loss plot
    plt.figure(figsize=(8, 6))
    plt.plot(iterations, critic_losses, 'r-')
    plt.title('Critic Losses')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(f"{plots_dir}/critic_losses.png", dpi=300)
    plt.close()
    
    # Returns plot
    plt.figure(figsize=(8, 6))
    plt.plot(iterations, returns, 'g-')
    plt.title('Average Returns')
    plt.xlabel('Iteration')
    plt.ylabel('Return')
    plt.grid(True)
    plt.savefig(f"{plots_dir}/returns.png", dpi=300)
    plt.close()
    
    # Entropy loss plot
    plt.figure(figsize=(8, 6))
    plt.plot(iterations, entropy_losses, 'm-')
    plt.title('Entropy Losses')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(f"{plots_dir}/entropy_losses.png", dpi=300)
    plt.close()