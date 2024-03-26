function environment(initial_state)
    # Initialize variables
    t = 1  # Time step counter
    
    # History vectors to store simulation results
    x_history = Vector{Float64}(undef, n_samples)  # System state history
    o_history = Vector{Float64}(undef, n_samples)  # Observations history
    u_applied_history = Vector{Float64}(undef, n_samples)  # Applied control actions history

    x = initial_state  # Current state of the system
    
    function state_transition(action)
        # Apply control action and update the system state with noise
        u_applied_history[t] = action
        x = (1-b*dt/m)*x + (dt/m)*action + randn() * sqrt(Q_NOISE)
        x_history[t] = x
    end
    
    function observe(noise_variance)
        # Generate an observation based on the current state with observation noise
        observation = x + randn() * sqrt(noise_variance)
        o_history[t] = observation
        return observation
    end
    function increment_time()
        # Increment the time step counter
        t += 1
    end
    function get_environment_history()
        # Returns the history of the system states, observations, and applied actions
        return x_history, o_history, u_applied_history
    end
    return state_transition, observe, increment_time, get_environment_history
end
