"""All public bath families through sixth-order preparation and propagation."""
import numpy as np
import pytest

from fasttcl import (
    BrownianBath, DiscreteBath, DrudeLorentzBath, ExponentialBath,
    PowerLawBath, SampledBath, SpectralDensityBath, compile_plan, prepare_model, solve,
)


@pytest.fixture(scope="module")
def model_and_plan():
    model = prepare_model(np.diag([-.5, .5]), [[0, .3], [.3, 0]])
    return model, compile_plan(model)


@pytest.mark.parametrize("bath_factory", [
    lambda: PowerLawBath(2., exponent_s=.5, temperature=.7),
    lambda: BrownianBath(omega_0=1.5, damping=.3, temperature=.7),
    lambda: DrudeLorentzBath(gamma=1., uv_cutoff=8., temperature=.7),
    lambda: DiscreteBath([.7, 1.3], [.5, .2], temperature=.7),
    lambda: ExponentialBath([.6, .4], [.3+1.2j, .3-1.2j]),
    lambda: SpectralDensityBath(lambda w: w*np.exp(-w/2), temperature=.7, omega_max=30.),
], ids=["power-law", "brownian", "drude-uv", "discrete", "exponential", "custom"])
def test_bath_catalog_solve_and_half_step_samples_agree(model_and_plan, bath_factory):
    model, plan = model_and_plan
    bath = bath_factory()
    dt, n_steps = .05, 4
    times = (dt/2)*np.arange(2*n_steps+1)
    sampled = SampledBath(bath.correlation(times), dt/2)
    settings = dict(dt=dt, n_steps=n_steps, order=6, coupling_strength=.3, plan=plan)
    result = solve(model, bath, [[0, 0], [0, 1]], **settings)
    reference = solve(model, sampled, [[0, 0], [0, 1]], **settings)
    assert set(result.trajectories) == {2, 4, 6}
    for order, trajectory in result.trajectories.items():
        np.testing.assert_allclose(trajectory, reference.trajectories[order], atol=1e-14, rtol=0)
        np.testing.assert_allclose(np.trace(trajectory, axis1=1, axis2=2), 1, atol=1e-13, rtol=0)
        np.testing.assert_allclose(trajectory, trajectory.conj().transpose(0, 2, 1), atol=1e-13, rtol=0)
