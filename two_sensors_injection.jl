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
