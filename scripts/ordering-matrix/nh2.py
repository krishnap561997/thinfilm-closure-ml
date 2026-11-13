from common import *
u=sp.expand(3*q/h**3*(h*z-z**2/2)); w=w_of(u)
ht=-Q[1]; qt=-dx(sp.Rational(6,5)*q**2/h)
Dw=sp.expand(dt(w,ht,qt)+u*dx(w)+w*sp.diff(w,z))
t=sp.Symbol('t',real=True)
pNH=sp.expand(sp.integrate(Dw.subs(z,t),(t,z,h)))
# A2-equation NH source: -2 int u d_x p_NH dz
J=sp.expand(-2*sp.integrate(u*dx(pNH),(z,0,h)))
# write as -2 q d_x(<p>) form? just report collected
print(sp.collect(J,[Q[3],H[3]],sp.factor))
# also q-eq NH source check: -int d_x p_NH = -d_x int pNH
Iq=sp.integrate(pNH,(z,0,h)); I=sp.integrate(z*Dw,(z,0,h)); print('check int p = int z Dw:',sp.simplify(Iq-I))
