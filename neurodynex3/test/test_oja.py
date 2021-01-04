def test_oja():
    """Test if Oja learns from a cloud"""
    import neurodynex3.ojas_rule.oja as oja
    nr_samples = 5
    cloud = oja.make_cloud(n=nr_samples, ratio=1.234, angle=56.789)
    wcourse = oja.learn(cloud)
    assert wcourse.shape == (nr_samples, 2),\
        "oja.learn(cloud) did not return shape=(nrSamples,2)"
