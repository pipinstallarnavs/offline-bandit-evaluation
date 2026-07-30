import unittest
from run import run
class Tests(unittest.TestCase):
 def test_estimators_are_finite(self):
  r=run(1); self.assertGreater(r['true_policy_accuracy'],.8); self.assertTrue(all(abs(r[k])<2 for k in ['ips','snips','doubly_robust']))
if __name__=='__main__':unittest.main()
