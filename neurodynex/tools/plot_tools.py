import matplotlib.pyplot as plt
import brian2 as b2


def plot_voltage_and_current_traces(voltage_monitor, current, title=None, firing_threshold=None,
                                    do_show_plot=True, legend_location=1):
    """plots voltage and current .

    Args:
        voltage_monitor (StateMonitor): recorded voltage
        current (TimedArray): injected current
        title (string, optional): title of the figure
        firing_threshold (Quantity, optional): if set to a value, the firing threshold is plotted.
        do_show_plot (bool): optional, default=True. Set it to False to suppress the blocking pyplot.show() call.
    """

    assert isinstance(voltage_monitor, b2.StateMonitor), "voltage_monitor is not of type StateMonitor"
    assert isinstance(current, b2.TimedArray), "current is not of type TimedArray"

    time_values_ms = voltage_monitor.t / b2.ms

    # current
    axis_c = plt.subplot(211)
    c = current(voltage_monitor.t, 0)
    max_current = max(current(voltage_monitor.t, 0))
    min_current = min(current(voltage_monitor.t, 0))
    margin = 1.05 * (max_current - min_current)
    plt.plot(voltage_monitor.t / b2.ms, c, "r", lw=2)
    if margin > 0.:
        plt.ylim((min_current - margin) / b2.amp, (max_current + margin) / b2.amp)
    # plt.xlabel("t [ms]")
    plt.ylabel("Input current [A] \n min: {0} \nmax: {1}".format(min_current, max_current))
    plt.grid()

    axis_v = plt.subplot(212)

    plt.plot(time_values_ms, voltage_monitor[0].v / b2.mV, lw=2)
    if firing_threshold is not None:
        plt.plot(
            (voltage_monitor.t / b2.ms)[[0, -1]],
            [firing_threshold / b2.mV, firing_threshold / b2.mV],
            "r--", lw=2
        )
    max_val = max(voltage_monitor[0].v)
    if firing_threshold is not None:
        max_val = max(max_val, firing_threshold)
    min_val = min(voltage_monitor[0].v)
    margin = 0.05 * (max_val - min_val)
    plt.ylim((min_val - margin) / b2.mV, (max_val + margin) / b2.mV)
    plt.xlabel("t [ms]")
    plt.ylabel("membrane voltage [mV]\n min: {0}\n max: {1}".format(min_val, max_val))
    plt.grid()

    if firing_threshold is not None:
        plt.legend(["vm", "firing threshold"], fontsize=12, loc=legend_location)

    if title is not None:
        plt.suptitle(title)
    if do_show_plot:
        plt.show()
