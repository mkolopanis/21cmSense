from unittest import TestCase

import py21cmsense as py21cm
import numpy as np
import os

class TestUtils(TestCase):

    def setUp(self):
        self.path = os.path.dirname(os.path.realpath(__file__))
        pass

    def tearDown(self):
        pass

    def test_load_no_files(self):
        out_f,_,_ = py21cm.utils.load_noise_files(None)
        self.assertEqual(0,out_f)

    def test_load_empty_list(self):
        out_f,_,_ = py21cm.utils.load_noise_files([])
        self.assertEqual(0,out_f)

    def test_load_ks(self):
        test_file = 'test_data/test_load_k_0.114.npz'
        ref_ks = np.linspace(.1,.3,50)
        print test_file
        _,out_k,_ = py21cm.utils.load_noise_files(
                os.path.join(self.path,test_file))
        print out_k
        self.assertTrue(np.allclose(ref_ks,out_k))

    def test_load_freq(self):
        test_file = 'test_data/test_load_k_0.114.npz'
        ref_freq=114
        out_freq,_,_ = py21cm.utils.load_noise_files(
                os.path.join(self.path,test_file))
        self.assertEqual(ref_freq,out_freq)

    def test_load_noise(self):
        test_file= 'test_data/test_load_k_0.114.npz'
        ref_noise = np.poly1d(np.polyfit([.1,.2,.3],[1,2,3],3))(np.linspace(.1,.3,50))
        _,_,out_noise = py21cm.utils.load_noise_files(
                os.path.join(self.path,test_file))
        # print out_noise
        # print np.load(os.path.join(self.path,test_file))['T_errs']
        self.assertTrue(np.allclose(ref_noise,out_noise))

    def test_flag_bad(self):
        test_file = 'test_data/test_data_0.180.npz'
        _,_,_,flags = py21cm.utils.load_noise_files(
            os.path.join(self.path,test_file),full=True
            )
        self.assertTrue(flags)

class TestInterp(TestCase):

    def setup(self):
        pass

    def tearDown(self):
        pass

    def test_no_freqs(self):
        out = py21cm.utils.noise_interp2d(None,[.1,.2,.3],[1,2,3])
        self.assertEqual(0,out)

    def test_no_ks(self):
        out = py21cm.utils.noise_interp2d(114,None,[1,2,3])
        self.assertEqual(0,out)

    def test_no_noises(self):
        out = py21cm.utils.noise_interp2d(114,[.1,.2,.3],None)
        self.assertEqual(0,out)

    def test_one_ks(self):
        with self.assertRaises(ValueError):
            py21cm.utils.noise_interp2d(114,[.1],[1])

    def test_two_ks(self):
        with self.assertRaises(ValueError):
            py21cm.utils.noise_interp2d(114,[.1,.1],[1,2])

    def test_two_one_ks(self):
        out=py21cm.utils.noise_interp2d(114,[[.1],[.1]],[[1],[2]])
        self.assertEqual(0,out)

    def test_interp_linear(self):
        out_interp=py21cm.utils.noise_interp2d([114,115],[[.1,.2],[.1,.2]],[[1,1],[2,2]])
        ref_num = 1.5
        out_num = out_interp(114.5,.15)
        self.assertEqual(ref_num,out_num)

if __name__ == '__main__':
        unittest.main()
