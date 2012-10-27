import unittest

import colorspace as cs

COLORS = {
  'indigo': {
    'HEX': '#4b0082',
    'RGB': [0.29412, 0.00000, 0.50980],
    'CIEXYZ': [0.06931, 0.03108, 0.21354],
    'CIEXYY': [0.22079, 0.09899, 0.03108],
    'CIELAB': [20.470, 51.695, -53.320],
    'CIELCH': [20.470, 74.265, 314.113],
    'CIELUV': [20.470, 10.084, -61.343],},
  'crimson': {
    'HEX': '#dc143c',
    'RGB': [0.86275, 0.07843, 0.23529],
    'CIEXYZ': [0.30581, 0.16042, 0.05760],
    'CIEXYY': [0.58380, 0.30625, 0.16042],
    'CIELAB': [47.030, 70.936, 33.595],
    'CIELCH': [47.030, 78.489, 25.342],
    'CIELUV': [47.030, 138.278, 19.641],},
  'white': {
    'HEX': '#ffffff',
    'RGB': [1, 1, 1],
    'CIEXYZ': [0.95050, 1.00000, 1.08900],
    'CIEXYY': [0.31272, 0.32900, 1],
    'CIELAB': [100, 0.005, -0.010],
    'CIELUV': [100, 0.001, -0.017],},
    # CIELCH omitted because Hue is almost completely},
    # irrelevant for white and its big rounding error
    # is acceptable here. Hue is better tested with 
    # more saturated colors, like the two above
  'black': {
    'HEX': '#000000',
    'RGB': [0, 0, 0],
    'CIEXYZ': [0, 0, 0],
    'CIEXYY': [0, 0, 0],
    'CIELAB': [0, 0, 0],
    'CIELUV': [0, 0, 0],
    # CIELCH omitted
    }
  }


PERMISSIBLE_ERROR = {
  'CIELAB': 0.01,
  'CIEXYZ': 0.001,
  'CIEXYY': 0.001,
  'CIELCH': 0.01,
  'CIELUV': 0.01,
  'RGB': 0.001,
}

class TestCs(unittest.TestCase):
    
    def te2st_simple_roundtrip(self):
        green = 0,1,0
        xyz = cs.CONV['RGB']['CIEXYZ'](*green)
        self.assertEquals(xyz, (0.3576, 0.7152, 0.1192))
        lab = cs.CONV['CIEXYZ']['CIELAB'](*xyz)
        self.assertEquals([round(x, 3) for x in lab], [87.737, -86.185, 83.181])
        lch = cs.CONV['CIELAB']['CIELCH'](*lab)
        self.assertEquals([round(x, 3) for x in lch], [87.737, 119.779, 136.016] )
        hcl = cs.CONV['RGB']['CIELCH'](*green)
        self.assertEquals([round(x, 3) for x in hcl], [87.737, 119.779, 136.016] )
        rgb = cs.CONV['CIELCH']['RGB'](*hcl)
        self.assertEquals([round(x, 3) for x in rgb], [0,1,0])
        # diff = big_dif(hcl, (0,0,0))
        # self.assertLessEqual(diff, PERMISSIBLE_ERROR[space2], 'conv:{0} to {1} input:{2} output:{3} should

    def t2est_conversion(self):
        for name, definitions in COLORS.items():
            for space1, data1 in definitions.items():
                for space2, data2 in definitions.items():
                    #try:
                    print "\nP", space1, space2, data2
                    output = cs.CONV[space1][space2](*data1)
                    print "OUTPUT!", output
                    # except TypeError as e:
                    #     print "  TE CONV", space1, space2, data1
                    #     print e
                    #     continue
                    # except KeyError as e:
                    #     print "  CONV", space1, space2, data1
                    #     print "KE", e
                    #     raise
                    if space2 == 'HEX':
                        print 'HEX conv:{0} to {1} input:{2} output:{3} shouldbe:{4} types:{5}{6}'.format(space1, space2, data1, output, data2, type(output), type(data2))
                        self.assertEquals(output, data2)
                        continue
                    diff = big_dif(output, data2)
                    self.assertLessEqual(diff, PERMISSIBLE_ERROR[space2], 'conv:{0} to {1} input:{2} output:{3} shouldbe:{4}'.format(space1, space2, data1, output, data2))

def big_dif(a,b):
    ret = 0
    for i in range(len(a)):
        dif = abs(a[i] - b[i])
        if dif > ret:
            ret = dif
    return ret

if __name__ == '__main__':
    unittest.main()
      
