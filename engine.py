import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import optuna
import logging
from scipy import integrate
# Disable Optuna's logging output
optuna.logging.set_verbosity(optuna.logging.WARNING)

"""
Definitions for descriptions:
i : Number of calibration points
j : Number of AP receivers
"""
class HataOkumura_Model:
  def __init__(self, ap_loc, wavelength = 0.12, M_bounds = [-300, 300], n_bounds = [1.75, 5.25], x_bounds = [-15, 15], y_bounds = [-15, 15], alpha = 0.45):
    self.ap_loc = ap_loc          # coordinates
    self.wavelength = wavelength
    self.M_lower = M_bounds[0]
    self.M_upper = M_bounds[1]
    self.n_lower = n_bounds[0]
    self.n_upper = n_bounds[1]
    self.x_lower = x_bounds[0]
    self.x_upper = x_bounds[1]
    self.y_lower = y_bounds[0]
    self.y_upper = y_bounds[1]
    self.alpha = alpha
  def HataOkumuraLogMean(self, P_RX, M, n):
    """
    INPUT
    P_RX  : Power level measured at the receivers in dBM
    M     :  𝐺𝑇 𝑋 − 𝐺𝑅𝑋 + 𝑃𝑇 𝑋 , where 𝐺𝑇 𝑋 , 𝐺𝑅𝑋 are the
            attenae gain of the transmitter and receiver respectively, in
            dBi, and 𝑃𝑇 𝑋 is the transmitted power level of the in dBm.
    n     : Obstruction coefficient

    RETURNS
    Mean of log-distance estimate from the receiver, using HataOkumura model
    """
    return ((M - P_RX + 20*np.log(self.wavelength) - 20*np.log(4*np.pi))/(10*n))
  def HataOkumuraLikelihood(self, dist, P_RX, M, n):
    """
    INPUT
    dist  : Suggested distance that we want to check likelihood for
    P_RX, M, n, wavelength : see HataOkumuraLogMean() function
    
    RETURNS
    Likelihood of seeing the received signal strength of P_RX, at the distance with the given parameters
    """
    est = self.HataOkumuraLogMean(P_RX, M, n)
    return norm.pdf(np.log(dist + 10**(-9)) - est, scale = self.alpha)
  def optuna_HataOkumura_find_coords_objective(self, trial,
                                        arrj_P_RX):
    """
    INPUT
    trial : used by optuna library for bayesian parameter search
    arrj_P_RX : (j x 1) numpy array of power level measured at the receivers in dBM

    OUTPUT
    Negative log likelihood of seeing the received signal strength of P_RX, at the distance with the given parameters
    """
    x = trial.suggest_float('x', self.x_lower, self.x_upper)
    y = trial.suggest_float('y', self.y_lower, self.y_upper)
    return self.HataOkumura_coords_NLL(x, y, arrj_P_RX)
  def find_coords_MLE(self, arrj_P_RX, n_trials = 250):
    """
    INPUT
    arrj_P_RX : (j x 1) numpy array of power level measured at the receivers in dBM
    n_trials  : Number of iterations to run bayesian parameter optimization search on
    
    RETURNS
    Maximum likelihood estimate of location
    """
    find_coords = optuna.create_study()
    find_coords.optimize(lambda trial : self.optuna_HataOkumura_find_coords_objective(trial, arrj_P_RX), n_trials=n_trials)
    return find_coords.best_params['x'], find_coords.best_params['y']
  def find_coords_integration(self, arrj_P_RX, step = 0.5):
    """
    INPUT
    arrj_P_RX : (j x 1) numpy array of power level measured at the receivers in dBM
    
    RETURNS
    Estimate of location based on approximate integration, by evaluating points over a grid
    """
    tot_weight, tot_xweight, tot_yweight = 0, 0, 0
    for x in np.arange(self.x_lower, self.x_upper + step, step):
      for y in np.arange(self.y_lower, self.y_upper + step, step):
        weight = np.exp(-1*self.HataOkumura_coords_NLL(x, y, arrj_P_RX))
        tot_weight += weight
        tot_xweight += weight*x
        tot_yweight += weight*y
    return tot_xweight/tot_weight, tot_yweight/tot_weight
 
### FOR UNIQUE M, n for each AP
class HataOkumura_independent_Mn_Model(HataOkumura_Model):
  def __init__(self, ap_loc, wavelength = 0.12, M_bounds = [-300, 300], n_bounds = [1.75, 5.25], x_bounds = [-15, 15], y_bounds = [-15, 15], alpha = 0.45):
    """
    INPUT
    ap_loc     : coordinates of the AP receiver (j x 2) numpy array
    wavelength : wavelength of signal in meters
    M/n bounds : upper/lower bounds for n (when calibrating)
    x/y bounds : used for parameter search and integration grid
    alpha      : standard deviation of log distance error estimate (for likelihood estimation)
    """
    super().__init__(ap_loc, wavelength, M_bounds, n_bounds, x_bounds, y_bounds, alpha)
    self.M = []
    self.n = []
  def optuna_HataOkumuraCalibrateObjective(self, trial, # input: M, n, alpha
                          label_x, label_y, arri_P_RX, ap_loc_x, ap_loc_y):
    """
    INPUT
    trial     : used for optimization with the optuna library.
    label_x   : x-coordinates of the GT labels, length n numpy array
    label_y   : y-coordinates of the GT labels, length n numpy array
    arri_P_RX : received signal strength of each receiver, length n numpy arrays
    ap_loc_x, ap_loc_y : x/y coordinates of the ap loc

    RETURNS
    Log likelihood estimate
    """
    M = trial.suggest_float('M', self.M_lower, self.M_upper)
    n = trial.suggest_float('n', self.n_lower, self.n_upper)
    s = 0
    for i in range(len(label_x)):
      x, y = label_x[i], label_y[i]
      dist = np.sqrt((x - ap_loc_x)**2 + (y - ap_loc_y)**2)
      s += np.log(self.HataOkumuraLikelihood(dist, arri_P_RX[i], M, n) + 10**(-9))
    # priors on n, M
    #s += np.log(norm.pdf(abs(M), scale = 500) + 10**(-5))
    #s += np.log(norm.pdf(abs(n - 3), scale = 2) + 10**(-5))
    return s

  def calibrate_Mn(self, label_x, label_y, arrij_P_RX, n_trials = 150):
    """
    INPUT
    label_x, label_y : x/y-coordinates of the GT labels, length n numpy array
    arrij_P_RX  : received signal strength of each receiver, (n x j) numpy arrays
    n_trials  : Number of iterations to run bayesian parameter optimization search on

    RETURNS
    Maximum likelihood estimate of model parameters M, n for each access point
    """
    Ms = []
    ns = []
    for j in range(len(self.ap_loc)):
      study = optuna.create_study(direction='maximize')
      study.optimize(lambda trial : self.optuna_HataOkumuraCalibrateObjective(trial, label_x, label_y, arrij_P_RX[:,j], self.ap_loc[j][0], self.ap_loc[j][1]), n_trials=n_trials)
      M = study.best_params['M']
      n = study.best_params['n']
      Ms.append(M)
      ns.append(n)
    self.Ms = Ms
    self.ns = ns
    return Ms, ns

  def HataOkumura_coords_NLL(self, x, y,# input: x, y
                            arrj_P_RX):
    """
    INPUT
    x     : x coordinate of the client location
    y     : y coordinate of the client location
    P_RX  : Power level measured at the receivers in dBM, length j numpy array
    wavelength : wavelength of signal in meters

    OUTPUT
    Negative log likelihood of seeing the received signal strength of P_RX, at the distance with the given parameters
    """
    s = 0
    for j in range(len(self.ap_loc)):
      dist = np.sqrt((x - self.ap_loc[j][0])**2 + (y- self.ap_loc[j][1])**2)
      s += np.log(self.HataOkumuraLikelihood(dist, arrj_P_RX[j], self.Ms[j], self.ns[j]) + 10**(-9))
    return -s
    
class HataOkumura_linked_Mn_Model(HataOkumura_Model):
  def __init__(self, ap_loc, wavelength = 0.12, M_bounds = [-300, 300], n_bounds = [1.75, 5.25], x_bounds = [-15, 15], y_bounds = [-15, 15], alpha = 0.45):
    """
    INPUT
    ap_loc     : coordinates of the AP receiver (j x 2) numpy array
    wavelength : wavelength of signal in meters
    M/n bounds : upper/lower bounds for n (when calibrating)
    x/y bounds : used for parameter search and integration grid
    alpha      : standard deviation of log distance error estimate (for likelihood estimation)
    """
    super().__init__(ap_loc, wavelength, M_bounds, n_bounds, x_bounds, y_bounds, alpha)
    self.M = 100
    self.n = 2.5

  def optuna_HataOkumuraCalibrateObjective(self, trial,
                          label_x, label_y, arrij_P_RX):
    """
    INPUT
    trial     : used for optimization with the optuna library.
    label_x   : x-coordinates of the GT labels, length n numpy array
    label_y   : y-coordinates of the GT labels, length n numpy array
    arri_P_RX : received signal strength of each receiver, length (n x j) numpy arrays

    RETURNS
    Log likelihood estimate
    """
    M = trial.suggest_float('M', self.M_lower, self.M_upper)
    n = trial.suggest_float('n', self.n_lower, self.n_upper)
    s = 0
    for i in range(len(label_x)):
      x, y = label_x[i], label_y[i]
      for j in range(len(self.ap_loc)):
        dist = np.sqrt((x - self.ap_loc[j][0])**2 + (y - self.ap_loc[j][1])**2)
        s += np.log(self.HataOkumuraLikelihood(dist, arrij_P_RX[i][j], M, n) + 10**(-9))
    # priors on n, M
    #s += np.log(norm.pdf(abs(M), scale = 500) + 10**(-5))
    #s += np.log(norm.pdf(abs(n - 3), scale = 2) + 10**(-5))
    return s
  def calibrate_Mn(self, label_x, label_y, arrij_P_RX, n_trials = 500):
    """
    INPUT
    label_x, label_y : x/y-coordinates of the GT labels, length n numpy array
    arrij_P_RX  : received signal strength of each receiver, (n x j) numpy arrays
    n_trials  : Number of iterations to run bayesian parameter optimization search on

    RETURNS
    Maximum likelihood estimate of model parameters M, n
    """
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial : self.optuna_HataOkumuraCalibrateObjective(trial, label_x, label_y, arrij_P_RX), n_trials=n_trials)
    M = study.best_params['M']
    n = study.best_params['n']
    self.M = M
    self.n = n
    return M, n
  def HataOkumura_coords_NLL(self, x, y,# input: x, y
                            arrj_P_RX):
    """
    INPUT
    x     : x coordinate of the client location
    y     : y coordinate of the client location
    arrj_P_RX  : Power level measured at the receivers in dBM, length j numpy array

    OUTPUT
    Negative log likelihood of seeing the received signal strength of P_RX, at the distance with the given parameters
    """
    s = 0
    for j in range(len(self.ap_loc)):
      dist = np.sqrt((x - self.ap_loc[j][0])**2 + (y- self.ap_loc[j][1])**2)
      s += np.log(self.HataOkumuraLikelihood(dist, arrj_P_RX[j], self.M, self.n) + 10**(-9))
    return -s

class LinearSignalDistance_Model:
  def __init__(self, ap_loc, x_bounds = [-15, 15], y_bounds = [-15, 15]):
    self.ap_loc = ap_loc
    self.x_lower = x_bounds[0]
    self.x_upper = x_bounds[1]
    self.y_lower = y_bounds[0]
    self.y_upper = y_bounds[1]
  def calibrate(self, label_x, label_y, arrij_P_RX):
    """
    INPUT
    label_x, label_y : x/y-coordinates of the GT labels, length n numpy array
    arrij_P_RX  : received signal strength of each receiver, (n x j) numpy arrays    
    """
    dists = []
    for i in range(len(label_x)):
      d = []
      for j in range(len(self.ap_loc)):
        d.append(np.sqrt((label_x[i] - self.ap_loc[j][0])**2 + (label_y[i] - self.ap_loc[j][1])**2))
      dists.append(d)
    dists = np.array(dists)
    all_errs = []
    # Estimate alpha via leave one out cross validation
    for i in range(len(label_x)):
      errors = []
      loo_dists = np.delete(dists, i, axis=0)
      loo_arrij_P_RX = np.delete(arrij_P_RX, i, axis=0)
      lr = LinearRegression()
      for j in range(len(self.ap_loc)):
        lr.fit(loo_arrij_P_RX[:,j].reshape(-1,1), loo_dists[:,j])
        pred = lr.predict(arrij_P_RX[i,j].reshape(-1,1))[0]
        errors.append(abs(pred - dists[i,j]))
      all_errs.append(errors)
    all_errs = np.array(all_errs)
    self.alphas = np.mean(all_errs, axis = 0)/np.sqrt(2/np.pi)
    self.models = []
    for j in range(len(self.ap_loc)):
      lr = LinearRegression()
      lr.fit(arrij_P_RX[:,j].reshape(-1,1), dists[:,j])
      self.models.append(lr)
  def NLL(self, x, y, arrj_P_RX):
    """
    INPUT
    x, y  : Coordinates we want to check likelihood for
    arrj_P_RX : (j x 1) numpy array of power level measured at the receivers in dBM

    RETURNS
    Likelihood of seeing the received signal strength of P_RX, at the distance with the given parameters
    """
    s = 0
    for j in range(len(self.ap_loc)):
      dist = np.sqrt((x - self.ap_loc[j][0])**2 + (y- self.ap_loc[j][1])**2)
      pred = self.models[j].predict(arrj_P_RX[j].reshape(-1,1))[0]
      s += norm.pdf(abs(dist - pred), scale = self.alphas[j])
    return -s
  def optuna_find_coords_objective(self, trial,
                                        arrj_P_RX):
    """
    INPUT
    trial : used by optuna library for bayesian parameter search
    arrj_P_RX : (j x 1) numpy array of power level measured at the receivers in dBM

    OUTPUT
    Negative log likelihood of seeing the received signal strength of P_RX, at the distance with the given parameters
    """
    x = trial.suggest_float('x', self.x_lower, self.x_upper)
    y = trial.suggest_float('y', self.y_lower, self.y_upper)
    return self.NLL(x, y, arrj_P_RX)
  def find_coords_MLE(self, arrj_P_RX, n_trials = 250):
    """
    INPUT
    arrj_P_RX : (j x 1) numpy array of power level measured at the receivers in dBM
    n_trials  : Number of iterations to run bayesian parameter optimization search on

    RETURNS
    Maximum likelihood estimate of location
    """
    find_coords = optuna.create_study()
    find_coords.optimize(lambda trial : self.optuna_find_coords_objective(trial, arrj_P_RX), n_trials=n_trials)
    return find_coords.best_params['x'], find_coords.best_params['y']
  def find_coords_integration(self, arrj_P_RX, step = 1):
    """
    INPUT
    arrj_P_RX : (j x 1) numpy array of power level measured at the receivers in dBM

    RETURNS
    Estimate of location based on approximate integration, by evaluating points over a grid
    """
    tot_weight, tot_xweight, tot_yweight = 0, 0, 0
    for x in np.arange(self.x_lower, self.x_upper + step, step):
      for y in np.arange(self.y_lower, self.y_upper + step, step):
        weight = np.exp(-1*self.NLL(x, y, arrj_P_RX))
        tot_weight += weight
        tot_xweight += weight*x
        tot_yweight += weight*y
    return tot_xweight/tot_weight, tot_yweight/tot_weight

###
# Save calibration coordinates to file
class Interface:
    def __init__(self):
        ## TODO - Write this ##
        self.ap_loc = []
        self.ap_ssid = []
        self.ap_macaddress = []
        self.model = None
        self.model_type = 'HataOkumura'
    def get_ap_loc(self):
        # reset 
        self.ap_loc = []
        self.ap_ssid = []
        self.ap_macaddress = []
        # load in data
        df = pd.read_csv("payload.csv")
        t0 = df[df['timestamp']==0].copy()
        pd.set_option('display.max_rows', None)
        print(t0)
        s = input("Select ap locations by typing their indices as comma separated values. ")
        indices = [int(x.strip()) for x in s.split(',')]
        indices = list(set(indices))
        for i in indices:
            self.ap_ssid.append(t0.iloc[i]['ssid'])
            self.ap_macaddress.append(t0.iloc[i]['macaddress'])
            coords = input(f"Enter coordinates x, y for {t0.iloc[i]['ssid']}, {t0.iloc[i]['macaddress']}: ")
            coords = [float(x.strip()) for x in coords.split(',')]
            self.ap_loc.append(coords)
        self.ap_loc = np.array(self.ap_loc)
    def calibrate(self):
        df = pd.read_csv("payload.csv")
        filtered = (df[df['macaddress'].isin(self.ap_macaddress)]).reset_index(drop=True)
        counts = filtered['timestamp'].value_counts()
        filtered = filtered[filtered['timestamp'].isin(counts[counts == len(self.ap_loc)].index)].reset_index(drop=True)
        print(filtered)
        s = input("Enter the indices for the timestamps to calibrate on: ")
        timestamps = [filtered.iloc[int(x.strip())]['timestamp'] for x in s.split(',')]
        timestamps = list(set(timestamps))
        ## TODO - Finish this function
        label_x, label_y, arrij_P_RX = [], [], []
        for t in timestamps:
            coords = input(f"Enter label coordinates x, y for timestamp {t}: ")
            coords = [float(x) for x in coords.split(',')]
            label_x.append(coords[0])
            label_y.append(coords[1])
            P_RX_vals = []
            for m in self.ap_macaddress:
                P_RX_vals.append(filtered[(filtered['timestamp'] == t) & (filtered['macaddress'] == m)]['rssi'].iloc[0])
            arrij_P_RX.append(P_RX_vals)
        arrij_P_RX = np.array(arrij_P_RX)
        s = input("Enter \'l\' to use experimental linear model (needs at least 2 calibration points)")
        if s == 'l':
            print("Using linear model")
            self.model = LinearSignalDistance_Model(self.ap_loc)
            self.model.calibrate(label_x, label_y, arrij_P_RX)
            self.model_type = 'linear'
        else:
            self.model_type = 'HataOkumura'
            s = input("Enter \'s\' to share parameters for each AP location: ")
            if s == 's':
                print("Using shared parameters")
                self.model = HataOkumura_linked_Mn_Model(self.ap_loc)
            else:
                print("Using independent parameters")
                self.model = HataOkumura_independent_Mn_Model(self.ap_loc)
            print("Calibrating...")
            self.model.calibrate_Mn(label_x, label_y, arrij_P_RX)
            print("Finished calibrating.")
        return
    def predict(self):
        df = pd.read_csv("payload.csv")
        filtered = (df[df['macaddress'].isin(self.ap_macaddress)]).reset_index(drop=True)
        counts = filtered['timestamp'].value_counts()
        filtered = filtered[filtered['timestamp'].isin(counts[counts == len(self.ap_loc)].index)].reset_index(drop=True)
        print(filtered)
        s = input("Enter the index for the timestamp to predict on: ")
        t = filtered.iloc[int(s.strip())]['timestamp']
        P_RX = []
        for m in self.ap_macaddress:
            P_RX.append(filtered[(filtered['timestamp'] == t) & (filtered['macaddress'] == m)]['rssi'].iloc[0])
        if self.model_type == 'HataOkumura': coords = self.model.find_coords_integration(P_RX)
        else: coords = self.model.find_coords_MLE(P_RX)
        print(f"Predicted client coordinates: {coords[0]:.2f}, {coords[1]:.2f}")
    def take_next_action(self):
        s = input("Enter \'p\' to make a new prediction, \'c\' to recalibrate, \'q\' to quit")
        if s == 'p':
            self.predict()
        if s == 'c':
            self.calibrate()
        return s
        
h = Interface()
df = pd.read_csv("payload.csv")
h.get_ap_loc()
h.calibrate()
while True:
    action = h.take_next_action()
    if action == 'q':
        break