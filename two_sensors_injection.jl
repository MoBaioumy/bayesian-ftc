# Define constants for the simulation
m = 100;             # Mass of the vehicle
dt = 0.01            # Time step
b = 5;               # Drag coefficient
OBS_NOISE = 0.1      # Observation noise variance
Q_NOISE = 0.01       # Process noise variance
const LARGE_VARIANCE = 1e10  # Define a large variance constant

# Initialize a new factor graph
g = FactorGraph()

# Define the prior state variables (mean and variance)
@RV x_min_mean
@RV x_min_var
@RV x_min ~ GaussianMeanVariance(x_min_mean, x_min_var)  # Prior state modeled as a Gaussian distribution

# Define the control input and process noise
@RV u                                         # Control input variable
@RV q ~ GaussianMeanVariance(0, Q_NOISE)      # Process noise 
@RV x = (1-b*dt/m)*x_min + (dt/m)*u + q       # Transition dynamics

# Define future control input and process noise for prediction
@RV u_future ~ GaussianMeanVariance(0, LARGE_VARIANCE)  
@RV q_future ~ GaussianMeanVariance(0, Q_NOISE)      # Future process noise, same as current
@RV x_future = (1-b*dt/m)*x + (dt/m)*u_future + q_future  # Future state prediction

# Define the mean and variance for the future state estimation
@RV x_future_mean
@RV x_future_var
@RV x_future ~ GaussianMeanVariance(x_future_mean, x_future_var)  # Future state modeled as a Gaussian distribution

# Define two observation variables derived from the current state `x`
@RV o_1 ~ GaussianMeanVariance(x, OBS_NOISE)  # First observation model with OBS_NOISE
@RV o_2 ~ GaussianMeanVariance(x, OBS_NOISE)  # Second observation model with OBS_NOISE

# Setup placeholders for injecting data into the model
placeholder(x_min_mean, :x_min_mean)
placeholder(x_min_var, :x_min_var)
placeholder(u, :u)
placeholder(x_future_mean, :x_future_mean)
placeholder(x_future_var, :x_future_var)
placeholder(o_1, :o_1)
placeholder(o_2, :o_2)

# Generate a message Passing Algo
algo = messagePassingAlgorithm([x; u_future]) # Figure out a schedule and compile to Julia code
algo_code = algorithmSourceCode(algo)
eval(Meta.parse(algo_code))

# Define an agent with an initial state (2 sensor case)
function agent(initial_state)
    t = 1  # Initialize the time step counter
    # Initialize the agent's belief about the initial state
    x_min_mean = initial_state  
    x_min_var = 100  
    
    # Prepare history containers for the agent's estimated state, variance, and future control actions
    x_mean_history = Vector{Float64}(undef, n_samples)
    x_var_history = Vector{Float64}(undef, n_samples)
    u_future_history = Vector{Float64}(undef, n_samples)

    # Define the inference function that updates the agent's beliefs based on new observations
    function infer(observation_1, observation_2, desired_state, last_control_action)
        # Prepare the data dictionary for inference
        data = Dict(
            :x_min_mean => x_min_mean,
            :x_min_var => x_min_var,
            :u => last_control_action, 
            :x_future_mean => desired_state,
            :x_future_var => 100,
            :o_1 => observation_1,
            :o_2 => observation_2
        )
        # Perform inference to update beliefs
        marginals = step!(data)
        
        # Update the agent's beliefs based on the inference results
        x_min_mean = mean(marginals[:x])
        x_min_var = var(marginals[:x])
        
        # Record the updated beliefs and the inferred future control action
        x_mean_history[t] = x_min_mean
        x_var_history[t] = x_min_var
        u_future_history[t] = mean(marginals[:u_future])
        
        # Increment the time step
        t += 1
        return u_future_history[t-1]
    end
    
    # Function to retrieve the agent's history of states, variances, and control actions
    function get_agent_history()
        return x_mean_history, x_var_history, u_future_history
    end
    return infer, get_agent_history
end


# Define the environment with 2 sensors
function environment(initial_state)
    t = 1  # Initialize the time step counter
    
    # Prepare history containers for the environment's true state and observations
    x_history = Vector{Float64}(undef, n_samples)
    o_1_history = Vector{Float64}(undef, n_samples)
    o_2_history = Vector{Float64}(undef, n_samples)
    u_applied_history = Vector{Float64}(undef, n_samples)

    x = initial_state  # Initialize the environment's state
    
    # Define the state transition function based on a control action
    function state_transition(action)
        u_applied_history[t] = action
        # Update the state based on the control action and process noise
        x = (1-b*dt/m)*x + (dt/m)*action + randn() * sqrt(Q_NOISE)
        x_history[t] = x
    end
    
    # Define the observation function for generating observations based on the current state
    function observe(noise_variance, sensor_set, injection_size=0)
        # Generate an observation based on the specified sensor and add it to the corresponding history
        observation = x + randn() * sqrt(noise_variance) + injection_size
        if sensor_set == 1
            o_1_history[t] = observation
        elseif sensor_set == 2
            o_2_history[t] = observation
        end
        return observation
    end
    
    # Function to increment the environment's time step
    function increment_time()
        t += 1
    end
    
    # Function to retrieve the environment's history of states and observations
    function get_environment_history()
        return x_history, o_1_history, o_2_history, u_applied_history
    end
    return state_transition, observe, increment_time, get_environment_history
end

# Define simulation parameters
n_samples = 100  # Number of simulation steps
desired_path = [i/10 for i=1:n_samples]  # Desired state values over time
initial_state = 0  # Starting state of the system
control_action = 0  # Initial control action

# Initialize the agent and environment with the initial state
infer, get_agent_history = agent(initial_state) 
state_transition, observe, increment_time, get_environment_history = environment(initial_state)  

# Make initial observations for both sensors
observation_1 = observe(OBS_NOISE, 1)  
observation_2 = observe(OBS_NOISE, 2)  

# Simulation loop
for t=1:n_samples
    # Infer the next control action based on current observations, desired state, and last action
    control_action = infer(observation_1, observation_2, desired_path[t], control_action)
    # Apply the control action to transition the environment state
    state_transition(control_action)
    # Update observations from sensor 1
    observation_1 = observe(OBS_NOISE, 1)
    # Conditionally update observations from sensor 2, introducing a change after 50 steps
    if t < 50
        observation_2 = observe(OBS_NOISE, 2)
    else
        observation_2 = observe(OBS_NOISE, 2, -10)  # Inject an anomaly or change in observation from sensor 2
    end
    # Increment the simulation time step
    increment_time()
end


# Retrieve simulation histories
x_history, o_1_history, o_2_history, u_applied_history = get_environment_history()
x_mean_history, x_var_history, u_future_history = get_agent_history();

# Plot desired path and actual state
plot(desired_path, "r--", label="Desired Velocity")
plot(x_history, "g--", label="Actual Velocity")

# Plot agent's estimated state
plot(x_mean_history, "b--", label="Estimated State")

# Plot observations from both sensors
plot(o_1_history, "r*", label="Sensor 1 Observations")
plot(o_2_history, "g*", label="Sensor 2 Observations")

# Display confidence interval around estimated state
fill_between(collect(1:n_samples), x_mean_history - sqrt.(x_var_history), 
    x_mean_history + sqrt.(x_var_history), color="b", alpha=0.2)

# Finalize plot
grid("on")
xlabel("t")
legend(loc="upper left");

# Calculate RMSE between the desired path and the actual/estimated states
rmse_actual = sqrt(mean((desired_path - x_history).^2))
rmse_estimated = sqrt(mean((desired_path - x_mean_history).^2))

println("RMSE Actual: $rmse_actual")
println("RMSE Estimated: $rmse_estimated")

# Save the plot to a file
savefig("simulation_results.png")

# Save simulation data to CSV or similar format for further analysis
using CSV, DataFrames
df = DataFrame(Time=1:n_samples, DesiredPath=desired_path, ActualVelocity=x_history, EstimatedState=x_mean_history)
CSV.write("simulation_data.csv", df)


