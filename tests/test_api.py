"""Physical interface, order selection, pictures, and propagation contracts."""
import importlib.abc
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from fasttcl import OhmicBath, SampledBath, prepare_model, generator_series, solve, compile_plan
from fasttcl import api


@pytest.fixture
def model():
    return prepare_model(np.diag([-.5, .5]), [[0, .3], [.3, 0]])


def test_lower_orders_do_not_import_sixth_or_taco():
    code = '''
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname == 'taco' or fullname.startswith(('taco.', 'fasttcl._engine', 'cupy')):
            raise AssertionError('unexpected dependency: '+fullname)
sys.meta_path.insert(0, Block())
from fasttcl import prepare_model, OhmicBath, solve
m=prepare_model([[0,.5],[.5,0]],[[.5,0],[0,-.5]])
for order in (2,4):
    r=solve(m,OhmicBath(2),[[1,0],[0,0]],dt=.02,n_steps=4,order=order,coupling_strength=.1)
    assert set(r.trajectories)==set(range(2,order+1,2))
'''
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_zero_coupling_is_unitary_without_tcl6_preparation(model, monkeypatch):
    monkeypatch.setattr(api, "compile_plan", lambda *a, **k: pytest.fail("unexpected compile"))
    rho0 = .5*np.ones((2, 2), dtype=complex)
    result = solve(model, OhmicBath(2), rho0, dt=.002, n_steps=50, coupling_strength=0, order=6)
    phase = np.exp(-1j*result.times[:, None]*model.energies)
    exact = phase[:, :, None]*rho0[None]*phase.conj()[:, None, :]
    for rho in result.trajectories.values():
        np.testing.assert_allclose(rho, exact, atol=2e-14, rtol=0)


def test_lambda_powers_and_cumulative_generators(model):
    a = generator_series(model, OhmicBath(2), dt=.05, n_steps=6, order=4, coupling_strength=.2)
    b = generator_series(model, OhmicBath(2), dt=.05, n_steps=6, order=4, coupling_strength=.4)
    for order in (2, 4):
        np.testing.assert_allclose(b.corrections[order], 2**order*a.corrections[order], atol=1e-15)
    np.testing.assert_allclose(a.cumulative(4), a.free+a.corrections[2]+a.corrections[4], atol=1e-15)
    with pytest.raises(ValueError):
        a.cumulative(6)


def test_sampled_and_callable_baths_agree(model):
    bath = OhmicBath(2)
    samples = bath.correlation(np.arange(17)*.025)
    a = solve(model, bath, [[1,0],[0,0]], dt=.05, n_steps=8, order=4, coupling_strength=.2)
    b = solve(model, SampledBath(samples, .025), [[1,0],[0,0]], dt=.05, n_steps=8, order=4, coupling_strength=.2)
    np.testing.assert_array_equal(a.rho, b.rho)
    with pytest.raises(ValueError, match="cover"):
        solve(model, SampledBath(samples[:8], .025), [[1,0],[0,0]], dt=.05, n_steps=8, order=4)
    with pytest.raises(ValueError, match="outside"):
        SampledBath(samples, .025).correlation([.0125])


def test_basis_covariance_for_complex_coupling():
    h = np.diag([-.7, .4, .8])
    a = np.array([[.1,.2+.1j,0],[.2-.1j,.05,.1j],[0,-.1j,-.15]])
    q, _ = np.linalg.qr(np.array([[1,2j,3],[2j,1,1j],[2,3,4j]], complex))
    rho = np.diag([1.,0,0])
    settings = dict(dt=.02,n_steps=4,coupling_strength=.2,order=4)
    original = solve(prepare_model(h,a), OhmicBath(2), rho, **settings)
    rotated = solve(prepare_model(q@h@q.conj().T,q@a@q.conj().T), OhmicBath(2), q@rho@q.conj().T, **settings)
    for k in (2,4):
        np.testing.assert_allclose(rotated.trajectories[k],q[None]@original.trajectories[k]@q.conj().T[None],atol=2e-14,rtol=0)


def test_actual_midpoint_generator_is_used(model, monkeypatch):
    def curved(m, bath, *, dt, n_steps, **kwargs):
        t = dt*np.arange(n_steps+1)
        diagonal = np.diag([0., -1., -1., 0.])
        return api.GeneratorSeries(t, np.zeros((4,4)), {2:t[:,None,None]**2*diagonal},m,1.,{})
    monkeypatch.setattr(api, "generator_series", curved)
    result = solve(model, OhmicBath(2), .5*np.ones((2,2)), dt=.1, n_steps=10, order=2)
    expected = .5*np.exp(-result.times**3/3)
    np.testing.assert_allclose(result.rho[:,0,1], expected, atol=3e-7,rtol=0)


@pytest.fixture(scope="module")
def sixth_fixture():
    m = prepare_model(np.diag([-.5,.5]), [[0,.2],[.2,0]])
    return m, compile_plan(m)


def test_sixth_order_scaling_and_physical_output(sixth_fixture):
    m, plan = sixth_fixture
    settings = dict(dt=.1,n_steps=4,order=6,plan=plan)
    x = generator_series(m,OhmicBath(2),coupling_strength=.2,**settings)
    y = generator_series(m,OhmicBath(2),coupling_strength=.4,**settings)
    assert np.max(np.abs(x.corrections[6])) > 1e-15
    np.testing.assert_allclose(y.corrections[6],64*x.corrections[6],rtol=1e-13,atol=1e-18)
    trace = np.array([1,0,0,1])
    swap = [0,2,1,3]
    for term in x.corrections.values():
        np.testing.assert_allclose(np.einsum('a,tab->tb',trace,term),0,atol=1e-13)
        np.testing.assert_allclose(term, term[:,swap][:,:,swap].conj(),atol=1e-13)
    result=solve(m,OhmicBath(2),[[0,0],[0,1]],dt=.05,n_steps=4,order=6,coupling_strength=.2,plan=plan)
    for r in result.trajectories.values():
        np.testing.assert_allclose(np.trace(r,axis1=1,axis2=2),1,atol=1e-13)
        np.testing.assert_allclose(r,r.conj().transpose(0,2,1),atol=1e-13)


@pytest.mark.parametrize("field,value",[("dt",0),("dt",np.nan),("dt",np.complex64(.1+2j)),("dt",np.bool_(True)),("n_steps",-1),("n_steps",1.5),("order",3),("order",True),("order",2.0),("coupling_strength",1j),("coupling_strength",np.complex64(.2+1j)),("coupling_strength",np.bool_(True))])
def test_reject_invalid_parameters(model,field,value):
    kwargs=dict(dt=.1,n_steps=4,order=2,coupling_strength=.2)
    kwargs[field]=value
    with pytest.raises((TypeError,ValueError)):
        generator_series(model,OhmicBath(2),**kwargs)


def test_zero_steps_returns_initial_state(model):
    result=solve(model,OhmicBath(2),[[1,0],[0,0]],dt=.1,n_steps=0,order=6)
    assert result.rho.shape==(1,2,2)
    np.testing.assert_allclose(result.rho[0],[[1,0],[0,0]],atol=1e-15)


@pytest.mark.parametrize("bad", [np.complex64(.1+2j), np.bool_(True), np.nan, 0])
def test_bath_scalar_validation(bad):
    with pytest.raises(ValueError):
        OhmicBath(bad)
    with pytest.raises(ValueError):
        SampledBath([1, .5], bad)


def test_gaussian_dephasing_matches_exact_solution():
    m = prepare_model(np.diag([-.5, .5]), np.diag([.5, -.5]))
    lam = .3
    result = solve(m, OhmicBath(2), .5*np.ones((2,2)),dt=.01,n_steps=100,order=6,coupling_strength=lam)
    expected = .5*np.exp(1j*result.times)*(1+4*result.times**2)**(-lam**2/4)
    np.testing.assert_allclose(result.rho[:,0,1],expected,atol=2.2e-7,rtol=0)
    np.testing.assert_allclose(result.generators.corrections[4],0,atol=1e-15)
    np.testing.assert_allclose(result.generators.corrections[6],0,atol=1e-15)
