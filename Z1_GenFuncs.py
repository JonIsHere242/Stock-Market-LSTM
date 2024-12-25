# data_preparation.py
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from typing import List, Optional, Dict, Callable
from tqdm import tqdm
import multiprocessing
from scipy.stats import skew, kurtosis
from PyEMD import EMD
from pyentrp import entropy as ent
import traceback
from networkx import Graph, clustering, degree_centrality
#from mfdfa import MFDFA  # You'd need to implement or import MFDFA
import scipy

from sklearn.neighbors import NearestNeighbors
from scipy.stats import skew, kurtosis
try:
    from pyrqa.time_series import TimeSeries
    from pyrqa.settings import Settings
    from pyrqa.computation import RQAComputation
    from pyrqa.analysis_type import Classic
    from pyrqa.metric import EuclideanMetric
    from pyrqa.neighbourhood import FixedRadius
    RQA_AVAILABLE = True
except ImportError:
    RQA_AVAILABLE = False




# Feature registry
FEATURE_FUNCTIONS: Dict[str, Callable] = {}

def register_feature(name: str):
    def decorator(func: Callable):
        FEATURE_FUNCTIONS[name] = func
        return func
    return decorator

@register_feature('time_reversal_asymmetry')
def compute_time_reversal_asymmetry(series: pd.Series, window: int) -> pd.Series:
    def asymmetry(x):
        x = np.asarray(x)
        if len(x) < 3:
            return np.nan
        return np.mean((x[2:] - x[1:-1])**2 * (x[1:-1] - x[:-2]))
    result = series.rolling(window=window, min_periods=3).apply(asymmetry, raw=False)
    return result

@register_feature('time_between_extremes')
def compute_time_between_extremes(price_series: pd.Series, window: int = None) -> pd.Series:
    """
    Compute time between price extremes. Window parameter is included but unused
    to maintain compatibility with other feature functions.
    """
    time_since_extreme = pd.Series(index=price_series.index, dtype=float)
    last_extreme_index = None
    last_extreme_type = None
    
    # Use expanding window to avoid lookahead bias
    rolling_max = price_series.expanding().max()
    rolling_min = price_series.expanding().min()
    
    for idx in price_series.index:
        price = price_series.loc[idx]
        if price == rolling_max.loc[idx]:
            if last_extreme_index is not None and last_extreme_type != 'high':
                time_since_extreme.loc[idx] = (idx - last_extreme_index).days
            last_extreme_index = idx
            last_extreme_type = 'high'
        elif price == rolling_min.loc[idx]:
            if last_extreme_index is not None and last_extreme_type != 'low':
                time_since_extreme.loc[idx] = (idx - last_extreme_index).days
            last_extreme_index = idx
            last_extreme_type = 'low'
        else:
            time_since_extreme.loc[idx] = np.nan if last_extreme_index is None else (idx - last_extreme_index).days
    
    return time_since_extreme.ffill()

@register_feature('volume_entropy')
def compute_volume_entropy(volume_series: pd.Series, window: int) -> pd.Series:
    entropy_series = pd.Series(index=volume_series.index, dtype=float)
    for i in range(window, len(volume_series)):
        data = volume_series.iloc[i - window:i]
        hist = np.histogram(data, bins='auto')[0]
        hist = hist / np.sum(hist)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        idx = volume_series.index[i]
        entropy_series.loc[idx] = entropy
    return entropy_series

@register_feature('information_rate_of_change')
def compute_information_rate_of_change(entropy_series: pd.Series, window: int) -> pd.Series:
    rate_of_change = entropy_series.diff().rolling(window=window).mean()
    return rate_of_change






@register_feature('emd_features')
def compute_emd_features(series: pd.Series, window: int, num_imfs: int = 3) -> pd.DataFrame:
    """Modified to be more robust"""
    result = pd.DataFrame(index=series.index)
    
    def compute_single_emd(data):
        try:
            emd = EMD()
            imfs = emd.emd(data)
            features = {}
            for j in range(min(num_imfs, len(imfs))):
                imf = imfs[j]
                features[f'IMF_{j+1}_Energy'] = np.sum(imf ** 2)
                features[f'IMF_{j+1}_Std'] = np.std(imf)
            return features
        except Exception:
            # Create a dictionary with NaN values for all IMF features
            features = {}
            for j in range(1, num_imfs + 1):
                features[f'IMF_{j}_Energy'] = np.nan
                features[f'IMF_{j}_Std'] = np.nan
            return features

    # Initialize columns
    for j in range(1, num_imfs + 1):
        result[f'IMF_{j}_Energy'] = np.nan
        result[f'IMF_{j}_Std'] = np.nan

    # Use rolling window with minimum periods
    for i in range(window, len(series)):
        data = series.iloc[i - window:i].values
        idx = series.index[i]
        features = compute_single_emd(data)
        for key, value in features.items():
            result.loc[idx, key] = value

    # Forward fill NaN values within each IMF
    for j in range(1, num_imfs + 1):
        cols = [f'IMF_{j}_Energy', f'IMF_{j}_Std']
        result[cols] = result[cols].ffill()

    return result







@register_feature('entropy')
def compute_entropy(series: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling entropy with robust handling of edge cases.
    """
    def safe_entropy(x):
        if len(x) < 2:
            return np.nan
        # Remove NaN values
        x = x[~np.isnan(x)]
        if len(x) < 2:
            return np.nan
        # Use robust binning
        hist, _ = np.histogram(x, bins='auto', range=(x.min(), x.max()))
        hist = hist / len(x)  # Normalize
        # Filter out zeros before log calculation
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))
    
    return series.rolling(window=window, min_periods=2).apply(safe_entropy)

@register_feature('noise')
def compute_noise_features(series: pd.Series, window: int) -> pd.DataFrame:
    white_noise = pd.Series(index=series.index, dtype=float)
    pink_noise = pd.Series(index=series.index, dtype=float)
    for i in range(window, len(series)):
        data = series.iloc[i - window:i].values
        idx = series.index[i]
        freqs = np.fft.fft(data)
        power = np.abs(freqs) ** 2
        white_noise.loc[idx] = np.std(power)
        pink_noise.loc[idx] = np.mean(power)
    return pd.DataFrame({'WhiteNoise': white_noise, 'PinkNoise': pink_noise})

@register_feature('rqa_features')
def compute_rqa_features(series: pd.Series, window: int) -> pd.DataFrame:
    if not RQA_AVAILABLE:
        print("RQA library is not available.")
        return pd.DataFrame()
    result = pd.DataFrame(index=series.index)
    for i in range(window, len(series)):
        data = series.iloc[i - window:i].values
        idx = series.index[i]
        try:
            time_series = TimeSeries(data)
            settings = Settings(time_series,
                                analysis_type=Classic,
                                neighbourhood=FixedRadius(0.1),
                                similarity_measure=EuclideanMetric(),
                                embedding_dimension=2,
                                time_delay=1)
            computation = RQAComputation.create(settings, verbose=False)
            rqa_result = computation.run()
            rqa_features = {
                'RecurrenceRate': rqa_result.recurrence_rate,
                'Determinism': rqa_result.determinism,
                'EntropyDiagonalLines': rqa_result.entropy_diagonal_lines,
                'Laminarity': rqa_result.laminarity,
                'TrappingTime': rqa_result.trapping_time,
            }
            for key, value in rqa_features.items():
                result.loc[idx, key] = value
        except Exception:
            continue
    return result


@register_feature('higuchi_fractal_dimension')
def compute_higuchi_fractal_dimension(series: pd.Series, window: int, k_max: int = 5) -> pd.Series:
    """
    Compute Higuchi's Fractal Dimension over a rolling window.
    
    Parameters:
    - series: pd.Series of price data
    - window: int, size of the rolling window
    - k_max: int, maximum number of intervals

    Returns:
    - pd.Series of fractal dimension values
    """
    def higuchi_fd(ts, kmax):
        try:
            N = len(ts)
            L = []
            x = ts.values
            for k in range(1, kmax + 1):
                Lk = []
                for m in range(k):
                    Lmk = 0
                    n_max = int(np.floor((N - m - 1) / k))
                    for n in range(1, n_max):
                        Lmk += abs(x[m + n * k] - x[m + (n - 1) * k])
                    Lmk = (Lmk * (N - 1)) / (k * n_max * k)
                    Lk.append(Lmk)
                L.append(np.mean(Lk))
            lnL = np.log(L)
            lnk = np.log(1.0 / np.arange(1, kmax + 1))
            # Linear fit
            fit = np.polyfit(lnk, lnL, 1)
            return fit[0]
        except Exception:
            return np.nan

    return series.rolling(window=window).apply(lambda x: higuchi_fd(x, k_max), raw=False)



@register_feature('lyapunov_exponent')
def compute_lyapunov_exponent(series: pd.Series, window: int, embedding_dim: int = 2, time_delay: int = 1) -> pd.Series:
    """Modified to handle small windows better"""
    def lyapunov_exp(ts):
        try:
            ts = ts.dropna().values
            if len(ts) < embedding_dim * time_delay:
                return np.nan
            # Use a simpler estimation method for small windows
            # Calculate divergence of nearby trajectories
            d0 = np.diff(ts)
            return np.log(np.abs(d0)).mean()
        except Exception:
            return np.nan

    return series.rolling(window=window).apply(lyapunov_exp, raw=False)



@register_feature('wavelet_energy')
def compute_wavelet_energy(series: pd.Series, window: int, wavelet: str = 'db1', level: int = 3) -> pd.DataFrame:
    """
    Compute wavelet energy at different decomposition levels over a rolling window.

    Parameters:
    - series: pd.Series of price data
    - window: int, size of the rolling window
    - wavelet: str, type of wavelet to use
    - level: int, number of levels for decomposition

    Returns:
    - pd.DataFrame with wavelet energies at each level
    """
    import pywt  # Make sure to install PyWavelets

    energies = {f'WaveletEnergy_Lv{lvl}': [] for lvl in range(1, level + 1)}
    indices = []

    for i in range(window, len(series)):
        data = series.iloc[i - window:i].values
        idx = series.index[i]
        try:
            coeffs = pywt.wavedec(data, wavelet=wavelet, level=level)
            for lvl, coeff in enumerate(coeffs[1:], start=1):  # Skip approximation coefficients
                energy = np.sum(coeff ** 2)
                energies[f'WaveletEnergy_Lv{lvl}'].append(energy)
            indices.append(idx)
        except Exception:
            for lvl in range(1, level + 1):
                energies[f'WaveletEnergy_Lv{lvl}'].append(np.nan)
            indices.append(idx)

    return pd.DataFrame(energies, index=indices)


@register_feature('dfa_alpha')
def compute_dfa(series: pd.Series, window: int) -> pd.Series:
    """
    Compute the scaling exponent (alpha) using Detrended Fluctuation Analysis over a rolling window.

    Parameters:
    - series: pd.Series of price data
    - window: int, size of the rolling window

    Returns:
    - pd.Series of DFA alpha values
    """
    from nolds import dfa  # Make sure to install the 'nolds' library

    def dfa_alpha(ts):
        try:
            ts = ts.dropna().values
            if len(ts) < 20:
                return np.nan
            return dfa(ts)
        except Exception:
            return np.nan

    return series.rolling(window=window).apply(dfa_alpha, raw=False)


@register_feature('spectral_entropy')
def compute_spectral_entropy(series: pd.Series, window: int, sampling_rate: float = 1.0) -> pd.Series:
    """
    Compute the spectral entropy over a rolling window with adjusted parameters.
    """
    from scipy.signal import welch
    
    def spectral_ent(ts):
        try:
            # Use window size as nperseg instead of default 256
            f, Pxx = welch(ts, fs=sampling_rate, nperseg=min(len(ts), window))
            # Normalize and compute entropy
            Pxx = Pxx / np.sum(Pxx)
            spectral_entropy = -np.sum(Pxx * np.log2(Pxx + 1e-10))
            return spectral_entropy
        except Exception:
            return np.nan
    
    return series.rolling(window=window, min_periods=window).apply(
        lambda x: spectral_ent(x.dropna()), raw=False
    )


@register_feature('autocorrelation')
def compute_autocorrelation(series: pd.Series, window: int, lag: int = 1) -> pd.Series:
    """
    Compute the autocorrelation with proper sample size checks.
    """
    def safe_autocorr(ts):
        if len(ts) < lag + 2:  # Need at least lag+2 samples
            return np.nan
        ts = ts.dropna()
        if len(ts) < lag + 2:
            return np.nan
        return ts.autocorr(lag=lag)

    return series.rolling(window=window, min_periods=lag+2).apply(safe_autocorr, raw=False)





@register_feature('partial_autocorrelation')
def compute_partial_autocorrelation(series: pd.Series, window: int, lag: int = 1) -> pd.Series:
    """
    Compute the partial autocorrelation with proper sample size checks.
    """
    from statsmodels.tsa.stattools import pacf_yw

    def safe_pacf(ts):
        if len(ts) < lag + 2:  # Need at least lag+2 samples
            return np.nan
        ts = ts.dropna()
        if len(ts) < lag + 2:
            return np.nan
        try:
            pacf_values = pacf_yw(ts, nlags=lag, method='mle')
            return pacf_values[lag]
        except:
            return np.nan

    return series.rolling(window=window, min_periods=lag+2).apply(safe_pacf, raw=False)



# New Feature: Permutation Entropy
@register_feature('permutation_entropy')
def compute_permutation_entropy(series: pd.Series, window: int, order: int = 3) -> pd.Series:
    """
    Compute the permutation entropy over a rolling window.

    Parameters:
    - series: pd.Series of price data
    - window: int, size of the rolling window
    - order: int, order of permutation entropy

    Returns:
    - pd.Series of permutation entropy values
    """
    from pyentrp import entropy as ent  # Make sure to install pyentrp library

    entropy_series = pd.Series(index=series.index, dtype=float)
    for i in range(window, len(series)):
        data = series.iloc[i - window:i].values
        idx = series.index[i]
        try:
            pe = ent.permutation_entropy(data, order=order, delay=1, normalize=True)
            entropy_series.loc[idx] = pe
        except Exception:
            entropy_series.loc[idx] = np.nan
    return entropy_series

@register_feature('hurst_exponent')
def compute_hurst_exponent(series: pd.Series, window: int) -> pd.Series:
    """
    Compute the Hurst exponent over a rolling window.

    Parameters:
    - series: pd.Series of price data
    - window: int, size of the rolling window

    Returns:
    - pd.Series of Hurst exponent values
    """
    def hurst(ts):
        try:
            ts = np.array(ts)
            if len(ts) < 20:
                return np.nan  # Need at least 20 data points

            # Remove NaN values from ts
            ts = ts[~np.isnan(ts)]
            if len(ts) < 20:
                return np.nan

            lags = range(2, 20)
            tau = []
            for lag in lags:
                diff = ts[lag:] - ts[:-lag]
                variance = np.var(diff)
                if variance > 0:
                    tau.append(np.sqrt(variance))
                else:
                    tau.append(np.nan)
            tau = np.array(tau)

            # Filter out invalid tau values
            valid = (~np.isnan(tau)) & (tau > 0)
            if valid.sum() < 2:
                return np.nan

            lags = np.array(lags)[valid]
            tau = tau[valid]

            # Compute logarithms
            log_lags = np.log(lags)
            log_tau = np.log(tau)

            # Check for infinite values
            if np.any(np.isinf(log_lags)) or np.any(np.isinf(log_tau)):
                return np.nan

            # Perform linear regression
            poly = np.polyfit(log_lags, log_tau, 1)
            hurst_exp = poly[0] * 2.0
            return hurst_exp
        except Exception as e:
            # Optionally, print the exception for debugging
            # print(f"Error computing Hurst exponent: {e}")
            return np.nan

    return series.rolling(window=window).apply(hurst, raw=False)


#@register_feature('mfdfa')
#def compute_mfdfa(series: pd.Series, window: int, q_range=(-5, 5), num_q=11) -> pd.DataFrame:
#    """
#    Compute Multifractal DFA scaling exponents
#    """
#    def mfdfa_local(ts, q_values):
#        try:
#            mfdfa = MFDFA(ts)
#            scaling_exponents = mfdfa.fit(q_values)
#            return scaling_exponents
#        except Exception:
#            return np.full(len(q_values), np.nan)
#            
#    q_values = np.linspace(q_range[0], q_range[1], num_q)
#    results = pd.DataFrame(index=series.index)
#    
#    for i in range(window, len(series)):
#        data = series.iloc[i-window:i].values
#        exponents = mfdfa_local(data, q_values)
#        for q, exp in zip(q_values, exponents):
#            results.loc[series.index[i], f'MFDFA_q{q:.1f}'] = exp
#            
#    return results


@register_feature('recurrence_network')
def compute_recurrence_network_features(series: pd.Series, window: int) -> pd.DataFrame:
    """
    Compute network measures from recurrence plots
    """
    def network_features(ts):
        try:
            # Create recurrence matrix (simplified)
            data = ts.reshape(-1, 1)
            dist = scipy.spatial.distance.pdist(data)
            dist = scipy.spatial.distance.squareform(dist)
            rec_mat = dist < np.mean(dist)  # Threshold
            
            # Create network
            G = Graph(rec_mat)
            
            return {
                'NetworkDensity': len(G.edges()) / (len(G.nodes()) * (len(G.nodes())-1)/2),
                'AvgClustering': clustering(G),
                'AvgCentrality': np.mean(list(degree_centrality(G).values()))
            }
        except Exception:
            return {'NetworkDensity': np.nan, 'AvgClustering': np.nan, 'AvgCentrality': np.nan}
            
    return pd.DataFrame([network_features(series.iloc[i-window:i].values) 
                        for i in range(window, len(series))], 
                        index=series.index[window:])





@register_feature('phase_space')
def compute_phase_space_features(series: pd.Series, window: int, embedding_dim: int = 3) -> pd.DataFrame:
    """
    Compute features from phase space reconstruction
    """
    
    def phase_features(ts, m):
        try:
            # Create time-delayed vectors
            tau = 1  # Could use mutual information to optimize
            n = len(ts) - (m-1)*tau
            vectors = np.zeros((n, m))
            for i in range(m):
                vectors[:, i] = ts[i*tau:i*tau + n]
                
            # Compute features
            nbrs = NearestNeighbors(n_neighbors=2).fit(vectors)
            distances, _ = nbrs.kneighbors(vectors)
            
            return {
                'PS_MeanNN': np.mean(distances[:, 1]),
                'PS_StdNN': np.std(distances[:, 1]),
                'PS_MaxNN': np.max(distances[:, 1])
            }
        except Exception:
            return {'PS_MeanNN': np.nan, 'PS_StdNN': np.nan, 'PS_MaxNN': np.nan}
            
    return pd.DataFrame([phase_features(series.iloc[i-window:i].values, embedding_dim) 
                        for i in range(window, len(series))],
                        index=series.index[window:])



@register_feature('higher_order_moments')
def compute_higher_order_moments(series: pd.Series, window: int) -> pd.DataFrame:
    """
    Compute skewness and kurtosis over a rolling window with proper sample size checks.
    """
    def safe_skew(x):
        if len(x) < 3:  # Need at least 3 samples for meaningful skewness
            return np.nan
        return skew(x)
    
    def safe_kurt(x):
        if len(x) < 4:  # Need at least 4 samples for meaningful kurtosis
            return np.nan
        return kurtosis(x)
    
    skewness = series.rolling(window=window, min_periods=3).apply(safe_skew, raw=False)
    kurt = series.rolling(window=window, min_periods=4).apply(safe_kurt, raw=False)
    return pd.DataFrame({'Skewness': skewness, 'Kurtosis': kurt})



def _cleanup_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """Clean up inf and nan values in the dataframe with minimum sample checks"""
    # Replace inf/-inf with max/min of finite values or 0 if all infinite
    for col in df.select_dtypes(include=[np.number]).columns:
        mask = np.isfinite(df[col])
        if mask.sum() >= 2:  # Need at least 2 samples for valid statistics
            max_val = df.loc[mask, col].max()
            min_val = df.loc[mask, col].min()
            df[col] = df[col].replace([np.inf, -np.inf], [max_val, min_val])
        else:
            df[col] = df[col].replace([np.inf, -np.inf], 0)

    # Fill remaining NaNs with 0
    df = df.fillna(0)
    return df



class DataPrep:
    def __init__(self, data_dir: str, min_history: int = 100, features: List[str] = None):
        self.data_dir = data_dir
        self.min_history = min_history
        # If no features specified, use all registered features
        self.features = features if features is not None else list(FEATURE_FUNCTIONS.keys())
        
    def _compute_features(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        df = df.copy()

        # Basic price features
        try:
            df['Return'] = df['Close'].pct_change()
            df['High_Low'] = df['High'] - df['Low']
            df['High_Close'] = df['High'] - df['Close']
            df['Low_Close'] = df['Low'] - df['Close']
        except Exception as e:
            print(f"Error computing basic features: {e}")

        # Compute registered features
        for feature_name in self.features:
            if feature_name in FEATURE_FUNCTIONS:
                try:
                    feature_func = FEATURE_FUNCTIONS[feature_name]
                    result = feature_func(df['Close'], window)

                    if isinstance(result, pd.DataFrame):
                        for col in result.columns:
                            df[f'{feature_name}_{col}'] = result[col]
                    else:
                        df[feature_name] = result

                except Exception as e:
                    print(f"Error computing feature {feature_name}: {e}")
                    continue
                
        # Compute target variable (percentage return)
        df['Target'] = df['Return'].shift(-1) * 100

        # Clean up features
        df = self._cleanup_features(df)

        # Remove the first and last 5 rows
        if len(df) > 10:
            df = df.iloc[5:-5]

        return df


    def _cleanup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean up inf and nan values in the dataframe"""
        # Replace inf/-inf with max/min of finite values or 0 if all infinite
        for col in df.select_dtypes(include=[np.number]).columns:
            mask = np.isfinite(df[col])
            if mask.any():
                max_val = df.loc[mask, col].max()
                min_val = df.loc[mask, col].min()
                df[col] = df[col].replace([np.inf, -np.inf], [max_val, min_val])
            else:
                df[col] = df[col].replace([np.inf, -np.inf], 0)

        # Fill remaining NaNs with 0
        df = df.fillna(0)
        return df


    # Inside the _process_file method of DataPrep class

    def _process_file(self, file_path: str) -> Optional[pd.DataFrame]:
        try:
            df = pq.read_table(file_path).to_pandas()
            if len(df) < max(self.min_history, 20):  # Ensure enough samples for statistics
                print(f"Insufficient samples in {file_path}")
                return None
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            df.set_index('Date', inplace=True)

            if len(df) < self.min_history:
                print(f"Insufficient initial data: {len(df)} rows < {self.min_history} minimum")
                return None

            # Compute features
            df = self._compute_features(df)

            # Remove warm-up period (first year of data)
            warmup_date = df.index[0] + pd.DateOffset(years=1)
            df = df[df.index >= warmup_date]

            # Handle NaN values using the new recommended methods
            # 1. Forward fill time-based features
            time_based_features = ['time_between_extremes']
            df[time_based_features] = df[time_based_features].ffill()  # Using ffill() instead of fillna(method='ffill')

            # 2. Fill technical features with median of their non-NaN values
            technical_features = [
                'volume_entropy', 'information_rate_of_change', 
                'noise_WhiteNoise', 'noise_PinkNoise',
                'higuchi_fractal_dimension', 'spectral_entropy',
                'dfa_alpha', 'hurst_exponent'
            ]
            df[technical_features] = df[technical_features].fillna(df[technical_features].median())

            # 3. Forward fill EMD features
            emd_features = [col for col in df.columns if 'emd_features' in col]
            df[emd_features] = df[emd_features].ffill()  # Using ffill() instead of fillna(method='ffill')

            # 4. Fill wavelet features with zeros
            wavelet_features = [col for col in df.columns if 'WaveletEnergy' in col]
            df[wavelet_features] = df[wavelet_features].fillna(0)

            # 5. Fill remaining essential columns
            essential_columns = ['Close', 'Return', 'Target']
            df[essential_columns] = df[essential_columns].ffill()  # Using ffill() instead of fillna(method='ffill')

            # Check remaining NaNs
            remaining_nans = df.isna().sum()
            if remaining_nans.any():
                print("\nRemaining NaN values after cleaning:")
                print(remaining_nans[remaining_nans > 0])

            if len(df) < self.min_history:
                print(f"Insufficient data after processing: {len(df)} rows < {self.min_history} minimum")
                return None

            return df

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            traceback.print_exc()
            return None









    def _validate_data(self, df: pd.DataFrame) -> bool:
        """Validate the input data meets basic requirements."""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        # Check for required columns
        if not all(col in df.columns for col in required_columns):
            print(f"Missing required columns. Found: {df.columns.tolist()}")
            return False

        # Check for non-negative values in price columns
        for col in ['Open', 'High', 'Low', 'Close']:
            if (df[col] <= 0).any():
                print(f"Found non-positive values in {col}")
                return False

        # Check for sufficient non-NaN values
        if df[required_columns].isna().sum().max() > len(df) * 0.5:  # More than 50% NaNs
            print("Too many NaN values in required columns")
            return False

        return True



    def prepare_dataset(self, files: Optional[List[str]] = None) -> pd.DataFrame:
        if files is None:
            files = [f for f in os.listdir(self.data_dir) if f.endswith('.parquet')]

        print(f"\nProcessing {len(files)} files...")
        with multiprocessing.Pool() as pool:
            dfs = list(tqdm(
                pool.imap(self._process_file, [os.path.join(self.data_dir, f) for f in files]),
                total=len(files),
                desc="Processing files"
            ))

        # Filter out None values
        dfs = [df for df in dfs if df is not None]
        if not dfs:
            raise ValueError("No valid DataFrames were generated")

        print("\nConcatenating datasets...")
        all_df = pd.concat(dfs, axis=0).reset_index()

        print("Shuffling data within dates...")
        all_df = all_df.groupby('Date', group_keys=False).apply(lambda x: x.sample(frac=1.0))

        # Print summary statistics
        print("\n=== Dataset Summary ===")
        print(f"Total samples: {len(all_df):,}")
        print(f"Date range: {all_df['Date'].min()} to {all_df['Date'].max()}")
        print(f"Trading days: {all_df['Date'].nunique():,}")
        print(f"Features generated: {len(all_df.columns)}")
        print("\nFeature list:")
        for col in sorted(all_df.columns):
            print(f"- {col}")

        return all_df
    



if __name__ == "__main__":
    import argparse

    def main():
        parser = argparse.ArgumentParser(description='Data Preparation')
        parser.add_argument('--data_dir', type=str, default='Data/PriceData', help='Directory containing the data files')
        parser.add_argument('--test_mode', action='store_true', help='Run in test mode with a subset of data')
        parser.add_argument('--features', nargs='+', help='List of features to compute', default=None)
        parser.add_argument('--min_history', type=int, default=100, help='Minimum history length')
        parser.add_argument('--sample_size', type=str, help='Number or percentage of files to process (e.g., "100" or "10%")')
        args = parser.parse_args()

        data_dir = args.data_dir
        test_mode = args.test_mode
        features = args.features
        min_history = args.min_history
        
        prep = DataPrep(data_dir, min_history=min_history, features=features)
        
        # Get list of all parquet files
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
        
        if test_mode:
            print("Running in test mode with a subset of data.")
            files = all_files[:10]  # Sample 10 files
        elif args.sample_size:
            if args.sample_size.endswith('%'):
                # Handle percentage
                percentage = float(args.sample_size.rstrip('%'))
                num_files = int(len(all_files) * (percentage / 100))
                files = all_files[:num_files]
                print(f"Processing {percentage}% of files ({num_files} files)")
            else:
                # Handle absolute number
                num_files = min(int(args.sample_size), len(all_files))
                files = all_files[:num_files]
                print(f"Processing {num_files} files")
        else:
            files = all_files
            print(f"Processing all {len(files)} files")

        dataset = prep.prepare_dataset(files=files)

        print(f"\nDataset created with {len(dataset)} samples.")
        print("\nSample of data:")
        with pd.option_context('display.max_columns', None, 'display.max_rows', 5):
            print(dataset.sample(5))

    main()
