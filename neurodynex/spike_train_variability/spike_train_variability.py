# Computer exercise for chapter 7 of Neuronal Dynamics book
# By: Parima Ahmadipour-2016

import matplotlib.pyplot as plt
import numpy as np


# This function generates a sample from a binomial distribution.
# ro * delta_t is the probability that the sample is 1.
def Binomial_SampleGenerator(ro, delta_t):
    a = np.random.rand()
    sample = 0.0
    if (a <= ro * delta_t):
        sample = 1.0
    return sample


# This function generates samples from exponential distribution.
def ExpDist_SampleGenerator(cdf_values, d_t):
    u = np.random.rand()
    i = np.argmin(np.abs(cdf_values - u))
    s = d_t * i
    return s


# This function returns discrete values of Cumulative Distribution
# Function of exponential distribution.
def CDF_values(ro, d_t, t):
    t_steps = np.arange(0.0, t, d_t)
    values = 1.0 - np.exp(-ro * t_steps)
    return values


# Part 1-a
# This function generates n spikes and plots the ISI histogram.
# The probability of firing in each time step is ro * delta_t
def forward_sampling(ro=30, delta_t=0.001, n_spikes=100000, n_bins=30, show_plt=True):
    """
    {ro: firing rate, hazard function in HZ
    delta_t:time steps in s
    n_spikes:number of spikes to be generates
    n_bins: number of bins for plotting the histogram
    }
    """

    spike_t = np.array([])
    i = 0
    while len(spike_t) < n_spikes:
        sample = Binomial_SampleGenerator(ro, delta_t)
        if sample == 1:
            spike_t = np.append(spike_t, i * delta_t)
        i += 1
    ISI = np.diff(spike_t)
    plt.hist(ISI, n_bins, histtype="step", linewidth=4,
             normed=True, label="Forward sampling")
    plt.show(show_plt)
    plt.legend()


# Part 1-b
def inverse_transform_sampling(ro=30, d_t=0.001, n_spikes=10000, n_bins=30, show_plt=True):
    """
    {ro: firing rate, hazard function in HZ
    d_t:
    n_spikes:number of spikes to be generates
    n_bins: number of bins for plotting the histogram
    }
    """

    ISI = np.array([])  # array containing Inter Spike Interval values
    cdf_values = CDF_values(ro, d_t, t=10)
    while len(ISI) <= n_spikes:
        s = ExpDist_SampleGenerator(cdf_values, d_t)
        ISI = np.append(ISI, [s])

    plt.hist(ISI, n_bins, histtype="bar", normed=True,
             label="Inverse transform sampling")
    plt.legend()
    plt.show(show_plt)


# Part 1-c
# This function plots both histograms of part 1-a and 1-b, in addition to
# the main distribution (exponential distribution)
def plots(ro, delta_t, d_t, d_t_pdf, n_spikes, n_bins, t=0.3):
    """
    {ro: firing rate, hazard function in HZ
    delta_t:time step for forward sampling in s
    d_t: time step used for sampling from CDF of exponential distribution in s
    d_t_pdf: time step used for sampling from PDF of exponential distribution in s
    n_spikes:number of spikes to be generates
    n_bins: number of bins for plotting the histogram
    t: time period during which exponential distribution is plotted, in s
    }
    """
    time = np.arange(0.0, t, d_t_pdf)
    prob = ro * np.exp(-ro * time)
    plt.plot(time, prob, "*-", linewidth=2,
             label="Analytical ISI distribution")
    plt.legend()
    plt.title("ISI distributions")
    plt.xlabel("Time (s)")
    plt.ylabel("Probability density")
    inverse_transform_sampling(ro, d_t, n_spikes, n_bins, False)
    forward_sampling(ro, delta_t, n_spikes, n_bins, False)
    plt.show()


# Part 2-a
def forward_sampling_with_refractoriness(ro=30, delta_abs=0.02,
                                         delta_t=0.001, n_spikes=100000, n_bins=30, show_plt=True):
    """
    {ro: firing rate, hazard function in HZ
    delta_abs: absolute refractory period in s
    delta_t:time steps in s
    n_spikes:number of spikes to be generates
    n_bins: number of bins for plotting the histogram
    }
    """

    spike_t = np.array([0.0])  # Array storing spike times in s
    print(len(spike_t))
    i = 0
    last_spike_t = 0.0  # Stores the time when the last spike has occured
    while len(spike_t) <= n_spikes:
        sample = Binomial_SampleGenerator(ro, delta_t)
        i += 1  # step counter
        if sample == 1.0:
            last_spike_t = i * delta_t + delta_abs + \
                last_spike_t  # update of last spike time
            # update of array storing spikes
            spike_t = np.append(spike_t, last_spike_t)
            i = 0
    print(spike_t)
    ISI = np.diff(spike_t)  # Array containing Inter Spike Interval
    plt.hist(ISI, n_bins, normed=True, linewidth=4,
             histtype="step", label="Forward sampling")
    plt.legend()
    plt.show(show_plt)


# Part 2-b
def inverse_transform_sampling_with_refractoriness(ro=30, delta_abs=0.02,
                                                   d_t=0.001, n_spikes=100000, n_bins=30, show_plt=True):
    """
    {ro: firing rate, hazard function in HZ
    delta_abs: absolute refractory period in s
    d_t:time step used for sampling from CDF of exponential distribution in s
    n_spikes:number of spikes to be generates
    n_bins: number of bins for plotting the histogram
    }
    """
    ISI = np.array([])  # Array storing Inter Spike Interval
    cdf_values = CDF_values(ro, d_t, t=10.0)
    while len(ISI) < n_spikes:
        s = ExpDist_SampleGenerator(cdf_values, d_t) + delta_abs
        ISI = np.append(ISI, [s])
    plt.hist(ISI, n_bins, normed=1, histtype="bar",
             label="Inverse transform sampling")
    plt.legend()
    plt.show(plt.show)


# Part 2-c
# This function plots both histograms of part 2-a and 2-b, in addition to
# the analytical ISI distribution
def plots_with_refractoriness(ro, delta_abs, delta_t, d_t, d_t_pdf, n_spikes, n_bins, t=0.3):
    """
    {ro: firing rate, hazard function in HZ
    delta_abs: absolute refractory period in s
    delta_t:time step for forward sampling in s
    d_t: time step used for sampling from CDF of exponential distribution in s
    d_t_pdf: time step used for sampling from PDF of exponential distribution in s
    n_spikes:number of spikes to be generates
    n_bins: number of bins for plotting the histogram
    t: time period during which analytical distribution is plotted, in s
    }
    """
    t1 = np.arange(0.0, delta_abs, delta_t)
    t2 = np.arange(0.0, t, d_t_pdf)
    p1 = np.zeros((delta_abs / delta_t))
    p2 = ro * np.exp(-ro * t2)
    # time points used in plotting analytical ISI distribution
    t = np.append(t1, t2 + delta_abs)
    # probability density points used in plotting analytical ISI distribution
    p = np.append(p1, p2)
    plt.plot(t, p, "*-", linewidth=2, label="Analytical ISI distribution")
    plt.legend()
    plt.title("ISI distributions with refractoriness")
    plt.xlabel("Time (s)")
    plt.ylabel("Probability density")
    forward_sampling_with_refractoriness(
        ro, delta_abs, delta_t, n_spikes, n_bins, False)
    inverse_transform_sampling_with_refractoriness(
        ro, delta_abs, d_t, n_spikes, n_bins, False)
    plt.show()


if __name__ == "__main__":
    # Test for part 1-a:
    forward_sampling(30, 0.01, 100000, 30)

    # Test for part 1-b:
    inverse_transform_sampling(ro=30, d_t=0.001, n_spikes=10000, n_bins=30)

    # Test for part 1-c:
    plots(30, 0.001, 0.001, 0.01, 10000, 30, 0.15)

    # Test for part 2-a
    forward_sampling_with_refractoriness(30, 0.2, 0.001, 100000, 30, True)

    # Test for part 2-b
    inverse_transform_sampling_with_refractoriness(
        30, 0.02, 0.001, 100000, 30, True)
    # Test for part 2-c
    plots_with_refractoriness(30, 0.02, 0.001, 0.001, 0.01, 10000, 30, 0.3)
