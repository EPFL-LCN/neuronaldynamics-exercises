def test_simulate_passive_cable():
    """ Test cable_equation.passive_cable.simulate_passive_cable """
    from neurodynex3.cable_equation import passive_cable
    import brian2
    # run simulation, access all default values
    voltage_monitor, cable_model = passive_cable.simulate_passive_cable(
        current_injection_location=passive_cable.DEFAULT_INPUT_LOCATION,
        input_current=passive_cable.DEFAULT_INPUT_CURRENT,
        length=passive_cable.CABLE_LENGTH,
        diameter=passive_cable.CABLE_DIAMETER,
        r_longitudinal=passive_cable.R_LONGITUDINAL,
        r_transversal=passive_cable.R_TRANSVERSAL,
        e_leak=passive_cable.E_LEAK, initial_voltage=passive_cable.E_LEAK,
        capacitance=passive_cable.CAPACITANCE,
        nr_compartments=3,
        simulation_time=3 * brian2.defaultclock.dt)
    assert voltage_monitor is not None
    assert cable_model is not None
