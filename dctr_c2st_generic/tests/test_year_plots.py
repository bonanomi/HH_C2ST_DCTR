"""Dependency-light checks for physical signed histogram and loader semantics."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from dctr_c2st_generic.plot_year_transfer import histogram, load_population


class YearPlotsTest(unittest.TestCase):
    def test_signed_sumw2(self):
        h, v = histogram([.2, .4, 1.2], [2, -1, 3], [0, 1, 2])
        np.testing.assert_equal(h, [1, 3])
        np.testing.assert_equal(v, [5, 9])
        after, variance = histogram([.2, .4, 1.2], np.array([2, -1, 3]) * .833, [0, 1, 2])
        np.testing.assert_allclose(after, h * .833)
        np.testing.assert_allclose(variance, v * .833**2)

    def test_flow(self):
        h, _ = histogram([-1, 3], [2, 3], [0, 1, 2], 'fold')
        np.testing.assert_equal(h, [2, 3])
        h, _ = histogram([-1, 3], [2, 3], [0, 1, 2], 'drop')
        np.testing.assert_equal(h, [0, 0])

    def test_physical_weight_config(self):
        cfg = dict(channels={'2mu':['2mu']}, channel_ids={'2mu':40}, region_ids={'dycr':3},
            years={'2025':dict(source={},data_processes={'Data':['data']},
                mc_processes={'DY':{'datasets':['dy'],'is_dy':True},'TT':['tt']},
                data_weights={'mode':'unit'},mc_weights={'mode':'product','fields':[{'column':'nominal'}]},
                dy_weight={'producer':'dy_correction_weight','column':'dy_correction_weight'})})
        def fake(c, side, channel):
            self.assertEqual(len(c['base_processes']['DY']['weights']['fields']), 2)
            self.assertEqual(len(c['base_processes']['TT']['weights']['fields']), 1)
            return pd.DataFrame({'event_id':[side],'weight_before':[-2 if side=='base' else 1]})
        with patch('dctr_c2st_generic.plot_year_transfer.load_side', fake):
            d, m = load_population(cfg, '2025', '2mu', 'dycr', ['x'])
            self.assertEqual(m.weight.iloc[0], -2)
            self.assertEqual(d.weight.iloc[0], 1)


if __name__ == '__main__':
    unittest.main()
