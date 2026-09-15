"""TCL6 regression tests against reference values from the research code.

Sparse fixture: exact fusion and early support specialization. Dense fixture:
source execution (fusion off). Both use domain completion off, NumPy, dt=.2,
Nt=5, E=[0,1], and C(t)=.7 exp(-.3|t|)[(n+1)exp(-1.2it)+n exp(1.2it)],
n=(exp(2.4)-1)^-1. They contain the full wrapped interaction-picture K6.
"""
import importlib.resources
import subprocess
import sys

import numpy as np
import pytest

from fasttcl._engine.runtime import compile_hr_plan, evaluate_hr_plan
from fasttcl._engine.general_model import EnergyBasisModel
from fasttcl._engine.spin_boson import DampedModeBath, prepare_two_level_model


def data_dir():
    return importlib.resources.files("fasttcl").joinpath("data")


def bath():
    return DampedModeBath(strength=.7, decay_rate=.3, mode_frequency=1.2, beta=2.)

_REFERENCES = {'dense': {'coupling_real': [[0.2, 0.5], [0.5, -0.2]],
           'coupling_imag': [[0.0, 0.1], [-0.1, 0.0]],
           'generator_real': [[[0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0]],
                              [[-2.3625580712952437e-07,
                                9.450255459737378e-08,
                                9.450255459737378e-08,
                                2.45677880519405e-07],
                               [9.269109885745937e-08,
                                -2.585826069935652e-08,
                                -4.8300426458552674e-08,
                                -9.638769328771584e-08],
                               [9.269109885745937e-08,
                                -4.8300426458552674e-08,
                                -2.585826069935652e-08,
                                -9.638769328771584e-08],
                               [2.3625580712952437e-07,
                                -9.450255459737378e-08,
                                -9.450255459737378e-08,
                                -2.45677880519405e-07]],
                              [[-1.3511866119767841e-06,
                                1.4458612787417511e-06,
                                1.4458612787417511e-06,
                                2.4088835913540083e-06],
                               [9.88813485753227e-07,
                                -2.462238125954113e-06,
                                3.603105669091648e-06,
                                -1.4149175524528004e-06],
                               [9.88813485753227e-07,
                                3.603105669091648e-06,
                                -2.462238125954113e-06,
                                -1.4149175524528004e-06],
                               [1.3511866119767841e-06,
                                -1.4458612787417511e-06,
                                -1.4458612787417511e-06,
                                -2.4088835913540083e-06]],
                              [[3.952637618055341e-05,
                                -5.994469442485074e-06,
                                -5.994469442485074e-06,
                                -4.017125915241244e-05],
                               [-6.489586282457872e-06,
                                -3.001807705514029e-05,
                                7.76650884267716e-05,
                                5.715452753048413e-06],
                               [-6.489586282457872e-06,
                                7.76650884267716e-05,
                                -3.001807705514029e-05,
                                5.715452753048413e-06],
                               [-3.952637618055341e-05,
                                5.994469442485073e-06,
                                5.994469442485073e-06,
                                4.017125915241244e-05]],
                              [[0.00027028768778234976,
                                -6.58411307319065e-05,
                                -6.58411307319065e-05,
                                -0.0003472271820803272],
                               [-4.000662428631096e-05,
                                -0.00018711096711667094,
                                0.0004991551563576137,
                                5.568064132548822e-05],
                               [-4.000662428631096e-05,
                                0.0004991551563576137,
                                -0.00018711096711667094,
                                5.568064132548822e-05],
                               [-0.00027028768778234976,
                                6.584113073190648e-05,
                                6.584113073190648e-05,
                                0.0003472271820803272]]],
           'generator_imag': [[[0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0]],
                              [[0.0, 9.631745716333412e-09, -9.631745716333412e-09, 0.0],
                               [-9.074088753495645e-09,
                                9.610948575747037e-10,
                                8.458434051796245e-09,
                                9.435263381733764e-09],
                               [9.074088753495645e-09,
                                -8.458434051796245e-09,
                                -9.610948575747037e-10,
                                -9.435263381733764e-09],
                               [0.0, -9.63174571633341e-09, 9.63174571633341e-09, 0.0]],
                              [[0.0, -5.742071536041658e-07, 5.742071536041658e-07, 0.0],
                               [-2.179736441262565e-06,
                                -5.988288321092031e-06,
                                -5.202178623672785e-06,
                                2.192287680173277e-06],
                               [2.179736441262565e-06,
                                5.202178623672785e-06,
                                5.988288321092031e-06,
                                -2.192287680173277e-06],
                               [0.0, 5.742071536041657e-07, -5.742071536041657e-07, 0.0]],
                              [[0.0, -8.25975735977726e-06, 8.25975735977726e-06, 0.0],
                               [-2.4329840585164832e-05,
                                -6.934843826620154e-05,
                                -3.0532360446523965e-05,
                                2.6793192977648773e-05],
                               [2.4329840585164832e-05,
                                3.0532360446523965e-05,
                                6.934843826620154e-05,
                                -2.6793192977648773e-05],
                               [0.0, 8.259757359777262e-06, -8.259757359777262e-06, 0.0]],
                              [[0.0, -4.708189846917459e-05, 4.708189846917459e-05, 0.0],
                               [-0.00012889358369786978,
                                -0.00039487256486474243,
                                3.2425523540388426e-05,
                                0.00015935445791731724],
                               [0.00012889358369786978,
                                -3.2425523540388426e-05,
                                0.00039487256486474243,
                                -0.00015935445791731724],
                               [0.0, 4.70818984691746e-05, -4.70818984691746e-05, 0.0]]]},
 'sparse': {'coupling_real': [[0.0, 0.5], [0.5, 0.0]],
            'coupling_imag': [[0.0, 0.1], [-0.1, 0.0]],
            'generator_real': [[[0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0]],
                               [[-2.044936512601392e-07, 0.0, 0.0, 2.1264850216217384e-07],
                                [0.0, 8.680677402971914e-09, -8.68489253473399e-09, 0.0],
                                [0.0, -8.68489253473399e-09, 8.680677402971914e-09, 0.0],
                                [2.044936512601392e-07, 0.0, 0.0, -2.1264850216217384e-07]],
                               [[-1.4147085032393576e-06, 0.0, 0.0, 2.262051477788744e-06],
                                [0.0, -1.4030818502077693e-06, 3.113879825901821e-06, 0.0],
                                [0.0, 3.113879825901821e-06, -1.4030818502077693e-06, 0.0],
                                [1.4147085032393576e-06, 0.0, 0.0, -2.262051477788744e-06]],
                               [[3.056895053573882e-05, 0.0, 0.0, -3.1145250787606435e-05],
                                [0.0, -2.343946541677578e-05, 5.5877356704558e-05, 0.0],
                                [0.0, 5.5877356704558e-05, -2.343946541677578e-05, 0.0],
                                [-3.056895053573882e-05, 0.0, 0.0, 3.1145250787606435e-05]],
                               [[0.00021045222777617568, 0.0, 0.0, -0.0002727466023504743],
                                [0.0, -0.00015002107627445774, 0.00035059682538443756, 0.0],
                                [0.0, 0.00035059682538443756, -0.00015002107627445774, 0.0],
                                [-0.00021045222777617568, 0.0, 0.0, 0.0002727466023504743]]],
            'generator_imag': [[[0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0]],
                               [[0.0, 0.0, 0.0, 0.0],
                                [0.0, 8.318323311474214e-10, 7.86604641310454e-10, 0.0],
                                [0.0, -7.86604641310454e-10, -8.318323311474214e-10, 0.0],
                                [0.0, 0.0, 0.0, 0.0]],
                               [[0.0, 0.0, 0.0, 0.0],
                                [0.0, -4.6280308512849095e-06, -3.7001433308277437e-06, 0.0],
                                [0.0, 3.7001433308277437e-06, 4.6280308512849095e-06, 0.0],
                                [0.0, 0.0, 0.0, 0.0]],
                               [[0.0, 0.0, 0.0, 0.0],
                                [0.0, -5.497376082838985e-05, -2.1195375117058553e-05, 0.0],
                                [0.0, 2.1195375117058553e-05, 5.497376082838985e-05, 0.0],
                                [0.0, 0.0, 0.0, 0.0]],
                               [[0.0, 0.0, 0.0, 0.0],
                                [0.0, -0.0003179785523602403, 2.64300797958494e-05, 0.0],
                                [0.0, -2.64300797958494e-05, 0.0003179785523602403, 0.0],
                                [0.0, 0.0, 0.0, 0.0]]]}}

@pytest.mark.parametrize("case,fusion", [("sparse", "exact"), ("dense", "off")])
def test_full_k6_matches_original_runtime(case, fusion):
    reference = _REFERENCES[case]
    coupling = np.array(reference["coupling_real"]) + 1j * np.array(reference["coupling_imag"])
    model = prepare_two_level_model(np.diag([0., 1.]), coupling)
    plan = compile_hr_plan(data_dir=data_dir(), model=model, fusion=fusion)
    result = evaluate_hr_plan(plan, nt=5, dt=.2, bath=bath())
    expected = np.array(reference["generator_real"]) + 1j * np.array(reference["generator_imag"])
    np.testing.assert_allclose(result.generator, expected, rtol=2e-11, atol=2e-17)
    assert result.generator.shape == (5, 4, 4)
    assert result.report["partial"] is False
    assert result.report["diagnostics"]["trace_preservation_max_abs_error"] < 1e-15
    assert result.report["diagnostics"]["realigned_hc_max_abs_error"] < 1e-15


def test_general_dimension_source_reuse_parity():
    coupling = np.array([[0., .4, 0.], [.4, 0., .3j], [0., -.3j, 0.]])
    model = EnergyBasisModel([0., .7, 1.9], coupling)
    plan = compile_hr_plan(data_dir=data_dir(), model=model)
    source = evaluate_hr_plan(plan, nt=4, dt=.15, bath=bath(), execution="source")
    reuse = evaluate_hr_plan(plan, nt=4, dt=.15, bath=bath(), execution="reuse")
    np.testing.assert_allclose(reuse.generator, source.generator, rtol=2e-11, atol=2e-17)
    assert source.generator.shape == (4, 9, 9)
    assert np.max(np.abs(source.generator)) > 1e-8


def test_all_engine_modules_import_without_optional_runtimes():
    # Run independently of this pytest process's already-imported modules.
    code = r"""
import importlib.abc, importlib, pkgutil, sys
class RejectOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'taco', 'cupy', 'matplotlib', 'jsonschema'}:
            raise AssertionError('unexpected optional dependency: '+fullname)
sys.meta_path.insert(0, RejectOptional())
import fasttcl._engine as engine
for entry in pkgutil.iter_modules(engine.__path__):
    importlib.import_module(engine.__name__+'.'+entry.name)
"""
    subprocess.run([sys.executable, "-c", code], check=True)
