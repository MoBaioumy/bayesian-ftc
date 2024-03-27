# Import packages
using ForneyLab   # For probabilistic modeling and inference
using LinearAlgebra
using Random
import Distributions: pdf, MvNormal, rand 
using PyPlot  

# Include agent and env logic
include("Agent.jl")
include("Environment.jl")

# Cruise controller example 
#  xdot = -b/m v + u/m 
#  y = v 

# Define constants for the model
m = 100;            # Mass of the vehicle
dt = 0.01           # Time step
b = 5;              # Drag coefficient
OBS_NOISE = 0.1     # Observation noise variance
Q_NOISE = 0.01      # Process noise variance
const LARGE_VARIANCE = 1e10  # Define a large variance constant
 

# Initialize a factor graph to represent the probabilistic model
g = FactorGraph()

# Define prior for the minimum state (previous state)
@RV x_min_mean      # Mean of the prior on the minimum state
@RV x_min_var       # Variance of the prior on the minimum state
@RV x_min ~ GaussianMeanVariance(x_min_mean, x_min_var) 

# Define the control input and process noise for the current state
@RV u               # Control input
@RV q ~ GaussianMeanVariance(0, Q_NOISE)  # Process noise with zero mean and Q_NOISE variance
@RV x = (1-b*dt/m)*x_min + (dt/m)*u + q  # Current state model incorporating dynamics and noise

# Define the control input and process noise for the future state
@RV u_future ~ GaussianMeanVariance(0, LARGE_VARIANCE)  
@RV q_future ~ GaussianMeanVariance(0, Q_NOISE)     # Future process noise, similar to the current process noise
@RV x_future = (1-b*dt/m)*x + (dt/m)*u_future + q_future  # Future state model

# Define the prior for the future state
@RV x_future_mean   # Mean of the prior on the future state
@RV x_future_var    # Variance of the prior on the future state
@RV x_future ~ GaussianMeanVariance(x_future_mean, x_future_var) 

# Define the observation model for the current state
@RV o ~ GaussianMeanVariance(x, OBS_NOISE)  # Observation model with Gaussian noise

# Placeholder nodes for data insertion
placeholder(x_min_mean, :x_min_mean)
placeholder(x_min_var, :x_min_var)
placeholder(u, :u)
placeholder(x_future_mean, :x_future_mean)
placeholder(x_future_var, :x_future_var)
placeholder(o, :o)


# Message passing algorithm
algo = messagePassingAlgorithm([x; u_future]) # Figure out a schedule and compile to Julia code
algo_code = algorithmSourceCode(algo)
eval(Meta.parse(algo_code))

# Set the number of simulation steps
n_samples = 100
# Generate a desired path for the agent to follow
desired_path = [i/10 for i=1:n_samples]
# Initialize the starting state and the first control action
initial_state = 0
control_action = 0

# Initialize the agent, env, and obs
infer, get_agent_history = agent(initial_state)
state_transition, observe, increment_time, get_environment_history = environment(initial_state)
observation = observe(OBS_NOISE)

# Main simulation loop
for t=1:n_samples
    # Infer the next control action, simulate the effect
    control_action = infer(observation, desired_path[t], control_action)
    state_transition(control_action)
    observation = observe(OBS_NOISE)
    # Increment the internal time step for both the agent and environment
    increment_time()
end

# Retrieve historical data from the environment and agent
x_history, o_history, u_applied_history = get_environment_history()
x_mean_history, x_var_history, u_future_history = get_agent_history();

# Plot the desired path as a dashed red line
plot(desired_path, "r--", label="Desired Velocity")
# Plot the actual velocity of the environment as a dashed green line
plot(x_history, "g--", label="Actual Velocity")
# Plot the agent's estimated state as a dashed blue line
plot(x_mean_history, "b--", label="Estimated State")
# Plot observations as blue stars
plot(o_history, "b*", label="State Observations")
# Shade the area between the estimated state ± its standard deviation to represent confidence interval
fill_between(collect(1:n_samples), x_mean_history - sqrt.(x_var_history), 
    x_mean_history + sqrt.(x_var_history), color="b", alpha=0.2)
# Enable grid for better readability
grid("on")
# Label the x-axis as 't' (time steps)
xlabel("t")
# Add a legend in the upper left corner to distinguish between plot elements
legend(loc="upper left");

