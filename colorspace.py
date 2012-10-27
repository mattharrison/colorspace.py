r"""
Supported Color Spaces

* RGB: Standard RGB, the color space used on the web. All values
  range between 0 and 1. Be careful, rounding errors can result in
  values just outside this range.

* CIEXYZ: One of the first mathematically defined color spaces. Values
  range between 0 and 0.95047, 1.0 and 1.08883 for X, Y and Z
  respectively. These three numbers together define the white point,
  which can be different depending on the chosen illuminant. The
  commonly used illuminant D65 was chosen for this project.

* CIEXYY: Normalized version of the above.

* CIELAB: A color space made for perceptual uniformity. Recommended
  for characterization of color surfaces and dyes. L is lightness,
  spans from 0 to 100. 

  (a*, negative values indicate green while positive values indicate
  magenta) and its position between yellow and blue (b*, negative
  values indicate blue and positive values indicate yellow). 

* CIELCH: A cylindrical representation of CIELAB. L is lightness (0
  back to 100 white), C is chroma (think saturation) (0 grey/center to
  100/color purity) and H is hue. H spans from 0 to 360.

* CIELUV: Another color space made for perceptual
  uniformity. Recommended for characterization of color displays.

* CIELCHUV: Same as CIELCH, but based on CIELUV.

* HEX: A representation of RGB.

>>> import colorspace
>>> colorspace.DEBUG = True
>>> start = HCL(0, 50, 70)
>>> end = HCL(0, 90, 20)
>>> for step in start.sequential(steps=5):
...     print "\nHCL", step, CIELCH_to_RGB(step.l, step.c, step.h)

>>> for step in start.diverging(90, steps=6):
...     print step

>>> rgb = 1,0,0 #255, 0, 0
>>> xyz = RGB_to_CIEXYZ(*rgb)
>>> xyz
>>> rgb2 = CIEXYZ_to_RGB(*xyz)
>>> rgb2
>>> luv = CIEXYZ_to_CIELUV(*xyz)
>>> luv
>>> xyz2 = CIELUV_to_CIEXYZ(*luv)
>>> xyz2
>>> rgb3 = CIEXYZ_to_RGB(*xyz2)
>>> rgb3

>>> f(0)
>>> f(1)
>>> f(99)
>>> f(100)

>>> f_inv(f(0))
0.0
>>> f_inv(f(1))
1.0
>>> round(f_inv(f(99)))
99.0
>>> round(f_inv(f(100)))
100.0

>>> round(f(f_inv(0)))
0.0
>>> round(f(f_inv(1)))
1.0
>>> round(f(f_inv(99)))
99.0
>>> round(f(f_inv(100)))
100.0

>>> REF_U
>>> REF_V

"""

import math

DEBUG=False

def debug_func(func):
    def wrapper3(*args, **kwargs):
        if DEBUG: print "CALLING", func, args
        res = func(*args, **kwargs)
        if DEBUG: print '\t result', res
        return res
    wrapper3.__name__ = func.__name__
    return wrapper3

# port of boronine / colorspaces.js to python

def dot_product(a,b):
    #print "AB", a, b
    assert len(a) == len(b)
    ret = 0
    for i in range(len(a)):
        ret += a[i] * b[i]
    return ret

# The D65 standard illuminant
REF_X = 0.95047
REF_Y = 1.0
REF_Z = 1.08883
REF_U = (4 * REF_X) / (REF_X + (15 * REF_Y) + (3 * REF_Z))
REF_V = (9 * REF_Y) / (REF_X + (15 * REF_Y) + (3 * REF_Z))

# CIE L*a*b* constants
LAB_E = 0.008856
LAB_K = 903.3

M = [
    [3.2406, -1.5372, -0.4986],
    [-0.9689, 1.8758,  0.0415],
    [0.0557, -0.2040,  1.0570],
    ]
M2 =[
    [0.4124, 0.3576, 0.1805],
    [0.2126, 0.7152, 0.0722],
    [0.0193, 0.1192, 0.9505],
    ]
# Used for Lab
def f(t):
    if t > LAB_E:
        return math.pow(t, 1.0/3)
    else:
        return 7.787 * t + 16.0/116

def f_inv(t):
  if math.pow(t, 3) > LAB_E:
      return math.pow(t, 3)
  else:
      return (116 * t - 16.0) / LAB_K


CONV = {}
SPACES =  ['CIEXYZ',
              'CIEXYY',
              'CIELAB',
              'CIELCH',
              'CIELUV',
              'CIELCHUV',
              'RGB',
              'HEX']
for space in SPACES:
    CONV[space] = {}

@debug_func
def CIEXYZ_to_RGB(*xyz):
    def from_linear(c):
        a = 0.055
        if c <= 0.0031308:
            return 12.92 * c
        else:
            return 1.055 * math.pow(c, 1/2.4) - 0.055
    r = from_linear(dot_product(M[0], xyz))
    g = from_linear(dot_product(M[1], xyz))
    b = from_linear(dot_product(M[2], xyz))
    #return [round(num) for num in r,g,b]
    #hackY???
    mins = [min(x,1.) for x in [r,g,b]]
    return [max(x,0.0) for x in mins]


CONV['CIEXYZ']['RGB'] = CIEXYZ_to_RGB

@debug_func
def RGB_to_CIEXYZ(*rgb):
    def to_linear(c):
        a = 0.055
        if c > 0.04045:
            return math.pow((c+a)/(1+a), 2.4)
        else:
            return c/12.92
    rgbl = [to_linear(x) for x in rgb]
    x = dot_product(M2[0], rgbl)
    y = dot_product(M2[1], rgbl)
    z = dot_product(M2[2], rgbl)
    return x,y,z


CONV['RGB']['CIEXYZ'] = RGB_to_CIEXYZ

@debug_func
def CIEXYZ_to_CIEXYY(*xyz):
    total = float(sum(xyz))
    x,y,z = xyz
    if total == 0:
        return 0,0,y
    return x/total, y/total, y


CONV['CIEXYZ']['CIEXYY'] = CIEXYZ_to_CIEXYY

@debug_func
def CIEXYY_to_CIEXYZ(*xyy):
    x,y,y2 = xyy
    if y == 0:
        return 0,0,0
    return (x*y2/float(y)), y2, (1-x-y)*y2/float(y)


CONV['CIEXYY']['CIEXYZ'] = CIEXYY_to_CIEXYZ

@debug_func
def CIEXYZ_to_CIELAB(*xyz):
    x,y,z = xyz
    fx = f(x/REF_X)
    fy = f(y/REF_Y)
    fz = f(z/REF_Z)
    l = 116*fy - 16
    a = 500*(fx-fy)
    b = 200*(fy-fz)
    return l,a,b


CONV['CIEXYZ']['CIELAB'] = CIEXYZ_to_CIELAB

@debug_func
def CIELAB_to_CIEXYZ(*lab):

    l,a,b = lab
    y2 = (l+16)/116.0
    z2 = y2 - b/200.0
    x2 = a / 500.0 + y2
    x = REF_X * f_inv(x2)
    y = REF_Y * f_inv(y2)
    z = REF_Z * f_inv(z2)
    return x,y,z


CONV['CIELAB']['CIEXYZ'] = CIELAB_to_CIEXYZ

@debug_func
def CIEXYZ_to_CIELUV(*xyz):
    x,y,z = xyz
    if all(var == 0 for var in xyz):
        # black
        return 0,0,0
    u2 = 4*x / (x+(15.0 * y) + (3.0 * z))
    v2 = 9*y / (x+(15.0 * y) + (3.0 * z))

    l = 116.0 * f(y/REF_Y) - 16
    # black will create a divide by zero error
    if l == 0:
        return 0,0,0
    u = 13 * l * (u2 - REF_U)
    v = 13 * l * (v2 - REF_V)
    return l, u, v


CONV['CIEXYZ']['CIELUV'] = CIEXYZ_to_CIELUV

@debug_func
def CIELUV_to_CIEXYZ(*luv):
    l,u,v = luv
    # black will create a divide by zero error
    if l == 0:
        return 0,0,0

    y2 = f_inv((l+16.0)/116.0)
    u2 = u/(13.0*l) + REF_U 
    v2 = v/(13.0*l) + REF_V
    y = y2 * REF_Y
    x = 0 - (9.0 * y * u2)/((u2 - 4) * v2 - u2 * v2)
    z = (9.0 * y - (15 * v2 * y) - (v2 * x))/(3*v2)
    return x,y,z


CONV['CIELUV']['CIEXYZ'] = CIELUV_to_CIEXYZ

@debug_func
def scalar_to_polar(*data):
    l, v1, v2 = data
    c = math.pow(math.pow(v1,2)+math.pow(v2,2), .5)
    h_rad = math.atan2(v2,v1)
    h = h_rad *360/2.0/math.pi
    if h < 0:
        h += 360
    #capping at 100 - bug?/feature?
    return min(l,100), min(c,100), h
    #return l, c, h

def copy_func(func, name):
    def wrapper5(*args, **kwargs):
        if DEBUG: print "FOO!", name
        return func(*args, **kwargs)
    wrapper5.__name__ = name
    return wrapper5
    

CIELAB_to_CIELCH = copy_func(scalar_to_polar, 'CIELAB_to_CIELCH')
CIELUV_to_CIELCHUV = copy_func(scalar_to_polar, 'CIELUV_to_CIELCHUV')


CONV['CIELAB']['CIELCH'] = CIELAB_to_CIELCH
CONV['CIELUV']['CIELCHUV'] = CIELUV_to_CIELCHUV

@debug_func
def polar_to_scaler(*data):
    l,c,h = data
    h_rad = h/360.0 * 2 * math.pi
    var1 = math.cos(h_rad) * c
    var2 = math.sin(h_rad) * c
    return l, var1, var2


CIELCH_to_CIELAB = copy_func(polar_to_scaler, 'CIELCH_to_CIELAB')
CIELCHUV_to_CIELUV = copy_func(polar_to_scaler, 'CIELCHUV_to_CIELUV')
CONV['CIELCH']['CIELAB'] = CIELCH_to_CIELAB
CONV['CIELCHUV']['CIELUV'] = CIELCH_to_CIELAB

def to_256(*data):
    data = [round(n,3) for n in data]
    for ch in data:
        assert 0 <= ch <= 1
    return [int(round(ch * 255)) for ch in data]

@debug_func
def RGB_to_HEX(*rgb):
    result = ['#']
    rgb = to_256(*rgb)
    result.extend( ['%02x'%x for x in rgb])
    result = ''.join(result)
    return result

CONV['RGB']['HEX'] = RGB_to_HEX

@debug_func
def HEX_to_RGB(*hex):
    hex = ''.join(hex)
    if hex.startswith('#'):
        hex = hex[1:]
    r = hex[:2]
    g = hex[2:4]
    b = hex[4:6]
    return [int(x,16)/255.0 for x in [r,g,b]]
CONV['HEX']['RGB'] = HEX_to_RGB
        
def iden_gen(space):
    def wrapper4(*args):
        if DEBUG: print "running IDEN", space
        if len(args) > 4:
            #RGB hack
            result = ''.join(args)
            if DEBUG: print result
            return result
        if DEBUG: print args
        return args
    return wrapper4


def identity(*args):
    if len(args) > 4:
        #RGB hack
        return ''.join(args)
    return args

def parent_to_child(p, child, parent):
    def wrapper1(*x):
        if DEBUG: print "CONVERTING", child, parent
        r1 = CONV[child][parent](*x)
        if DEBUG: print "\t",r1
        if DEBUG: print "RUNNING P"
        r2 = p(*r1)
        if DEBUG: print "\t",r2
        return r2
    wrapper1.__name__ = '{0}_to_{1} WRAPPER'.format(child,parent)
    return wrapper1

def from_to_parent(p, child, parent):
    def wrapper2(*x):
        if DEBUG: print "RUNNING1", p
        r1 = p(*x)
        if DEBUG: print "\t",r1
        if DEBUG: print "CONV2", parent, child
        r1 = CONV[parent][child](*r1)
        if DEBUG: print "\t",r1
        return r1
    wrapper2.__name__ = '{0}_to_{1} WRAPPER'.format(child,parent)
    return wrapper2

def converter(from_,to_):
    tree = [
        ['CIELCH', 'CIELAB'],
        ['CIELCHUV', 'CIELUV'],
        ['HEX', 'RGB'],
        ['CIEXYY', 'CIEXYZ'],
        ['CIELAB', 'CIEXYZ'],
        ['CIELUV', 'CIEXYZ'],
        ['RGB', 'CIEXYZ'],
        ]
    def path(tree, from2, to2):
        if from2 == to2:
            #return identity #lambda *x:x
            return iden_gen(from2) #lambda *x:x
        child, parent = tree[0]
        if from2 == child:
            p = path(tree[1:], parent, to2)
            return parent_to_child(p, child, parent)
            #return lambda *x: p(*CONV[child][parent](*x))
        if to2 == child:
            p = path(tree[1:], from2, parent)
            return from_to_parent(p, child, parent)
            #return lambda *x: CONV[parent][child](*p(*x))
        p = path(tree[1:], from2, to2)
        return p
    func = path(tree, from_, to_)
    return func
                               



for space in SPACES:
    for space2 in SPACES:
        if space in CONV and space2 not in CONV[space]:
            name = '{0}_to_{1}'.format(space, space2)
            func =  converter(space,space2)
            locals()[name] = func
            CONV[space][space2] = func
                               
    

class Color(object):
    pass

class RBG(Color):
    def __init__(self, r, g, b):
        self.r = r
        self.b = b
        self.g = g

def float_range(start, end, step):
    num = start
    while num <= end:
        yield num
        num += step


class HCL(Color):
    """CIEHUV"""
    def __init__(self, h, c, l):
        # assert 0 <= h <= 360
        # assert 0 <= c <= 100
        # assert 0 <= l <= 100
        self.h = h
        self.c = c
        self.l = l

    def __str__(self):
        return 'HCL({0},{1},{2})'.format(self.h, self.c, self.l)

    def _delta(self, steps, c_target, l_target):
        c_delta = (c_target - self.c)/float(steps)
        l_delta = (l_target - self.l)/float(steps)
        c_next = self.c
        l_next = self.l
        for i in range(steps):
            c_next += c_delta
            l_next += l_delta
            yield HCL(self.h, c_next, l_next)
            
    def darker(self, steps):
        for x in self._delta(steps, c_target=0, l_target=0):
            yield x
    def lighter(self, steps):
        for x in self._delta(steps, c_target=0, l_target=100):
            yield x
        

    def darker2(self, steps):
        "move c and l toward 0"
        c_delta = self.c/float(steps)
        l_delta = self.l/float(steps)
        c_next = self.c
        l_next = self.l
        for i in range(steps):
            c_next -= c_delta
            l_next -= l_delta
            yield HCL(self.h, c_next, l_next)

    def diverging(self, steps, end_hue=None, l_max=99, l_min=40):
        """
        Go from saturated to lighter and back to saturated
        """
        if end_hue is None:
            end_hue = (self.h + 180)%360
        for c in reversed(list(self.sequential(steps/2, l_max=l_max,l_min=l_min))):
            yield c
        c_end = HCL(end_hue, c.c, c.l)
        for d in c_end.sequential(steps/2, #self.c, self.l,
                                  l_min=l_min):
            yield d
        

    def sequential(self, steps, c_max=100, l_max=90, l_min=30, i_gen=None):
        """ 
        Goes from light to dark
        l_max, and l_min default to avoid white (100) and black (0)
        """
        if i_gen is None:
            i_gen = float_range(0, 1, 1.0/(steps-1))
        c_delta = c_max - self.c
        for i in i_gen:
            assert 0<=i<=1
            c_next = self.c + i*c_delta
            l_next = l_max - i*(l_max-l_min)
            yield HCL(self.h, c_next, l_next)

    def qualitative(self, steps):
        h = self.h
        for i in range(steps):
            yield HCL(h, self.c, self.l)
            h = (h+(360.0/steps)) %360

class HSV(Color):
    """ Problematic because brightness is not consistent over hues and
    saturations. """
    def __init__(self, h, s, v):
        assert 0 <= h <= 360
        assert 0 <= s <= 100
        assert 0 <= v <= 100
        self.h = h
        self.s = s
        self.v = v


def test():
    import doctest
    doctest.testmod()


if __name__ == '__main__':
    test()


    
