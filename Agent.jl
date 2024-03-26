function agent(initial_state)
    # Initialize variables
    t = 1  # Time step counter
    x_min_mean = initial_state  # Initial mean of the state
    x_min_var = 100  # Initial variance of the state
    
    # History vectors to store simulation results
    x_mean_history = Vector{Float64}(undef, n_samples)  # State mean history
    x_var_history = Vector{Float64}(undef, n_samples)  # State variance history
    u_future_history = Vector{Float64}(undef, n_samples)  # Future control actions history

    function infer(observation, desired_state, last_control_action)
        # Prepare data for the inference step
        data = Dict(
            :x_min_mean => x_min_mean,
            :x_min_var => x_min_var,
            :u => last_control_action, 
            :x_future_mean => desired_state,
            :x_future_var => 100,
            :o => observation
        )
        # Perform inference step (this function, `step!`, should be defined elsewhere)
        marginals = step!(data)
        
        # Update internal state based on inference results
        x_min_mean = mean(marginals[:x])
        x_min_var = var(marginals[:x])
        
        # Update history
        x_mean_history[t] = x_min_mean
        x_var_history[t] = x_min_var
        u_future_history[t] = mean(marginals[:u_future])
        
        # Increment time step
        t += 1
        return u_future_history[t-1]
    end
    
    function get_agent_history()
        # Returns the history of the agent's internal states and actions
        return x_mean_history, x_var_history, u_future_history
    end
    return infer, get_agent_history
end
