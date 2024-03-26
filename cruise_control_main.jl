# Import packages
using ForneyLab   # For probabilistic modeling and inference
using LinearAlgebra
using Random
import Distributions: pdf, MvNormal, rand 
using PyPlot  


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
